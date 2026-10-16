"""Agent 层测试：结构化输出的容错、规划结果的可执行性约束、失败语义。

用假网关注入固定响应，因此**不调用真实模型、不消耗 token、结果确定**。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.agents import AgentContext, AgentOutputError, PlannerAgent, RetrieverAgent  # noqa: E402
from multiagent.agents.base import extract_json  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.contracts import (  # noqa: E402
    AgentError,
    AgentInvocation,
    AgentResult,
    InvocationBudget,
    ResultStatus,
)
from multiagent.gateway import LLMResponse  # noqa: E402
from multiagent.graph.pipeline import AnalysisPipeline, PipelineNodeFailed  # noqa: E402
from multiagent.tools import SqlSandbox  # noqa: E402

settings = Settings(_env_file=ROOT / ".env")


class FakeGateway:
    """按顺序返回预设内容的假网关。"""

    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)
        self.calls = 0

    async def complete(self, messages, *, max_tokens=None, temperature=None) -> LLMResponse:
        self.calls += 1
        if not self._contents:
            raise AssertionError("假网关的预设响应已用尽")
        return LLMResponse(
            content=self._contents.pop(0),
            provider="fake",
            model="fake-model",
            input_tokens=100,
            output_tokens=200,
            finish_reason="stop",
        )


def _ctx(gateway) -> AgentContext:
    return AgentContext(settings=settings, gateway=gateway, sandbox=SqlSandbox(settings))


def _invocation(**inputs) -> AgentInvocation:
    return _node_invocation("planner_1", "planner", **inputs)


def _node_invocation(node_id: str, role: str, **inputs) -> AgentInvocation:
    return AgentInvocation(
        task_id="11111111-2222-4333-8444-555555555555",
        node_id=node_id,
        attempt=1,
        agent_role=role,
        trace_id="trace-1",
        input={"question": "分析上月华东区销售额下滑原因", "schema": "- sales(...)", "now": "2026-10-16T12:00:00+08:00", **inputs},
        budget=InvocationBudget(max_tokens=8192, timeout_seconds=300),
    )


def _dag_json(node_ids_roles: list[tuple[str, str, list[str]]]) -> str:
    nodes = [
        {
            "node_id": nid,
            "name": f"节点 {nid}",
            "agent_role": role,
            "depends_on": deps,
            "inputs": {},
            "outputs": {},
            "acceptance": ["给出至少一个含具体数字的结论"],
        }
        for nid, role, deps in node_ids_roles
    ]
    import json

    return json.dumps(
        {
            "task_id": "11111111-2222-4333-8444-555555555555",
            "goal": "分析华东区销售额下滑原因",
            "created_at": "2026-10-16T12:00:00+08:00",
            "nodes": nodes,
        },
        ensure_ascii=False,
    )


# --------------------------------------------------------------- extract_json


def test_extract_json_plain() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fence() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_tolerates_surrounding_prose() -> None:
    assert extract_json('好的，结果如下：\n{"a": 1}\n希望对你有帮助。') == {"a": 1}


def test_extract_json_raises_on_garbage() -> None:
    with pytest.raises(AgentOutputError):
        extract_json("这不是 JSON")


# ------------------------------------------------------------------- Planner


def test_planner_accepts_executable_two_node_plan() -> None:
    gateway = FakeGateway(
        _dag_json([("retriever_1", "retriever", []), ("analyst_1", "analyst", ["retriever_1"])])
    )
    import asyncio

    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.ok
    assert result.outputs["node_count"] == 2
    assert [n["node_id"] for n in result.outputs["dag"]["nodes"]] == ["retriever_1", "analyst_1"]
    assert result.confidence is None
    assert result.tokens_used.total == 300


def test_planner_rejects_nodes_the_chain_cannot_execute() -> None:
    """规划了 reviewer_1 但链路不执行它 —— 必须报错，不能静默丢弃。"""
    gateway = FakeGateway(
        _dag_json(
            [
                ("retriever_1", "retriever", []),
                ("analyst_1", "analyst", ["retriever_1"]),
                ("reviewer_1", "reviewer", ["analyst_1"]),
            ]
        )
    )
    import asyncio

    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "UNSUPPORTED_PLAN"
    assert "reviewer_1" in result.error.message


def test_planner_retries_once_on_invalid_output() -> None:
    gateway = FakeGateway(
        "抱歉，我无法输出 JSON",  # 第一次：非法
        _dag_json([("retriever_1", "retriever", []), ("analyst_1", "analyst", ["retriever_1"])]),
    )
    import asyncio

    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.ok
    assert gateway.calls == 2, "首次解析失败应回灌错误重试一次"
    assert result.tokens_used.total == 600, "两次调用的 token 都要计入"


def test_planner_fails_after_exhausting_attempts() -> None:
    gateway = FakeGateway("不是 JSON", "还是不是 JSON")
    import asyncio

    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "SCHEMA_MISMATCH"
    assert result.error.retryable is True


def test_planner_reports_bad_input_without_calling_model() -> None:
    gateway = FakeGateway()
    inv = _invocation()
    inv.input.pop("question")
    import asyncio

    result = asyncio.run(PlannerAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "BAD_INPUT"
    assert result.error.retryable is False
    assert gateway.calls == 0, "输入不合法时不应浪费一次模型调用"


def test_planner_surfaces_graph_validation_errors() -> None:
    """依赖不存在的节点：SubtaskDAG 构造期就会拒绝，Planner 应回报为 SCHEMA_MISMATCH。"""
    gateway = FakeGateway(
        _dag_json([("retriever_1", "retriever", []), ("analyst_1", "analyst", ["ghost_1"])]),
        _dag_json([("retriever_1", "retriever", []), ("analyst_1", "analyst", ["ghost_1"])]),
    )
    import asyncio

    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "SCHEMA_MISMATCH"


# --------------------------------------------------------------- 错误码映射


@pytest.mark.parametrize(
    ("exc", "expected_code", "expected_retryable"),
    [
        (AgentOutputError("x"), "SCHEMA_MISMATCH", True),
        (ValueError("x"), "INTERNAL_ERROR", False),
    ],
)
def test_classify_error_mapping(exc, expected_code, expected_retryable) -> None:
    code, retryable = PlannerAgent.classify_error(exc)
    assert (code, retryable) == (expected_code, expected_retryable)


# --------------------------------------------------- 流水线的失败中止语义


def test_absorb_raises_on_failed_node_result() -> None:
    pipeline = AnalysisPipeline(settings)
    inv = _invocation()
    failed = AgentResult(
        task_id=inv.task_id,
        node_id="analyst_1",
        attempt=1,
        status=ResultStatus.failed,
        tokens_used={"input": 1, "output": 1, "total": 2},
        duration_ms=5,
        error=AgentError(code="SQL_ERROR", message="permission denied", retryable=False),
    )
    with pytest.raises(PipelineNodeFailed) as excinfo:
        pipeline._absorb({"blackboard": {}, "node_results": []}, inv, failed)
    assert excinfo.value.code == "SQL_ERROR"
    assert excinfo.value.retryable is False


def test_absorb_writes_blackboard_entry_on_success() -> None:
    pipeline = AnalysisPipeline(settings)
    inv = _node_invocation("retriever_1", "retriever")
    ok = AgentResult(
        task_id=inv.task_id,
        node_id="retriever_1",
        attempt=1,
        status=ResultStatus.ok,
        tokens_used={"input": 10, "output": 20, "total": 30},
        duration_ms=12,
        outputs={"metric_definitions": {"revenue": {}}},
        confidence=1.0,
        sources=[{"type": "sql", "ref": "sha256:abc"}],
        blackboard_writes=["retriever_1.metric_definitions"],
    )
    merge = pipeline._absorb({"blackboard": {}, "node_results": []}, inv, ok)
    assert "retriever_1.metric_definitions" in merge["blackboard"]
    entry = merge["blackboard"]["retriever_1.metric_definitions"]
    assert entry["version"] == 1
    assert entry["written_by"] == "retriever@0.1.0"
    assert merge["metric_definitions"] == {"revenue": {}}


# ------------------------------------------------- Retriever（需要数据库则跳过）


def test_retriever_returns_metric_definitions_from_db() -> None:
    import asyncio

    gateway = FakeGateway()  # 本节点不应调用模型
    try:
        result = asyncio.run(RetrieverAgent().invoke(_invocation(), _ctx(gateway)))
    except Exception as exc:
        pytest.skip(f"数据库不可达: {exc}")
    if result.status is ResultStatus.failed:
        pytest.skip(f"数据库不可达: {result.error.message}")
    assert result.status is ResultStatus.ok
    defs = result.outputs["metric_definitions"]
    assert "revenue" in defs and "refund_rate" in defs
    assert gateway.calls == 0, "口径是结构化数据，直查即可，不该额外调用模型"
    assert result.sources[0].type.value == "sql"
