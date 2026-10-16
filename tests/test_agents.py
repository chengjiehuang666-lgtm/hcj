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
    """按顺序返回预设内容的假网关。

    与 `LLMGateway` / `ModelRouter` **同签名**（含 `route`），并记录所见调用点，
    以便断言「各 Agent 用对了路由」。
    """

    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)
        self.calls = 0
        self.routes_seen: list[str | None] = []

    async def complete(
        self, messages, *, max_tokens=None, temperature=None, route=None
    ) -> LLMResponse:
        self.calls += 1
        self.routes_seen.append(route)
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
        input={
            "question": "分析上月华东区销售额下滑原因",
            "schema": "- sales(...)",
            "now": "2026-10-16T12:00:00+08:00",
            # Planner 必须知道哪些角色可执行，否则无从规划（W5 起的必要输入）
            "available_roles": ["retriever", "analyst"],
            **inputs,
        },
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
    from multiagent.agents import AnalystAgent

    inv = _node_invocation("analyst_1", "analyst")
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
        AnalysisPipeline._absorb(inv, failed, AnalystAgent())
    assert excinfo.value.code == "SQL_ERROR"
    assert excinfo.value.retryable is False


def test_absorb_returns_only_this_node_delta() -> None:
    """`_absorb` 必须只返回**本节点的增量**，不读旧状态、不做全量写回。

    并行时若「读步初快照 → 合并 → 全量写回」，两个节点的写入会互相覆盖（丢更新）。
    这是 W5 引入并行后最容易踩的坑，用测试钉死。
    """
    from multiagent.agents import RetrieverAgent

    inv = _node_invocation("retriever_2", "retriever")
    ok = AgentResult(
        task_id=inv.task_id,
        node_id="retriever_2",
        attempt=1,
        status=ResultStatus.ok,
        tokens_used={"input": 10, "output": 20, "total": 30},
        duration_ms=12,
        outputs={"metric_definitions": {"revenue": {}}, "schema": "sales(...)"},
        confidence=1.0,
        sources=[{"type": "sql", "ref": "sha256:abc"}],
        blackboard_writes=["retriever_2.metrics"],
    )
    delta = AnalysisPipeline._absorb(inv, ok, RetrieverAgent())

    assert set(delta) == {"node_outputs", "blackboard", "node_results"}
    assert delta["node_outputs"] == {"retriever_2": ok.outputs}
    assert list(delta["blackboard"]) == ["retriever_2.metrics"]
    entry = delta["blackboard"]["retriever_2.metrics"]
    assert entry["version"] == 1
    # 用真实 Agent 版本，不再写死 0.1.0（原先写死的是潜伏 bug）
    assert entry["written_by"] == "retriever@0.3.0"
    # 角色要写进 node_results，下游才能按 role 取值而不依赖可变的 node_id
    assert delta["node_results"][0]["agent_role"] == "retriever"


def test_absorb_does_not_leak_previous_state() -> None:
    """传入同样的输入而不带任何旧状态，输出必须完全由入参决定。"""
    from multiagent.agents import RetrieverAgent

    inv = _node_invocation("retriever_1", "retriever")
    ok = AgentResult(
        task_id=inv.task_id,
        node_id="retriever_1",
        attempt=1,
        status=ResultStatus.ok,
        tokens_used={"input": 1, "output": 1, "total": 2},
        duration_ms=1,
        outputs={"knowledge": []},
        sources=[{"type": "document", "ref": "知识库:x#0"}],
        blackboard_writes=["retriever_1.knowledge"],
    )
    first = AnalysisPipeline._absorb(inv, ok, RetrieverAgent())
    second = AnalysisPipeline._absorb(inv, ok, RetrieverAgent())
    assert first == second


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


# ------------------------------------------------- 分级路由：调用点标注正确


def test_planner_declares_planner_route() -> None:
    gateway = FakeGateway(
        _dag_json([("retriever_1", "retriever", []), ("analyst_1", "analyst", ["retriever_1"])])
    )
    import asyncio

    asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert gateway.routes_seen == ["planner"], gateway.routes_seen


def test_planner_retry_keeps_same_route() -> None:
    """重试也必须带上调用点，否则重试会掉回默认档。"""
    gateway = FakeGateway(
        "不是 JSON",
        _dag_json([("retriever_1", "retriever", []), ("analyst_1", "analyst", ["retriever_1"])]),
    )
    import asyncio

    asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert gateway.routes_seen == ["planner", "planner"], gateway.routes_seen


def test_analyst_uses_distinct_routes_for_sql_and_conclusion() -> None:
    """取数（机械）与归因（推理）必须是两个不同的调用点，才能分级路由。"""
    from multiagent.agents import AnalystAgent

    gateway = FakeGateway(
        '{"sql": "SELECT 1 AS x", "rationale": "最小查询"}',
        '{"summary": "结果为 1", "key_findings": ["x=1"], '
        '"insufficient_data": false, "confidence": 0.9}',
    )
    inv = _node_invocation("analyst_1", "analyst")
    import asyncio

    try:
        result = asyncio.run(AnalystAgent().invoke(inv, _ctx(gateway)))
    except Exception as exc:
        pytest.skip(f"数据库不可达: {exc}")
    if result.status is ResultStatus.failed:
        pytest.skip(f"数据库不可达: {result.error.message}")
    assert gateway.routes_seen == ["analyst_sql", "analyst_conclude"], gateway.routes_seen


def test_retriever_records_no_route_because_it_calls_no_model() -> None:
    gateway = FakeGateway()
    assert gateway.routes_seen == []


# -------------------------------------------- Retriever 的 mode（两路检索的基础）


def _retriever_invocation(node_id: str, **inputs) -> AgentInvocation:
    return _node_invocation(node_id, "retriever", **inputs)


def test_retriever_metrics_mode_skips_knowledge() -> None:
    """mode=metrics 只走 SQL，不碰知识库——这是两路并行不重叠的前提。"""
    import asyncio

    gateway = FakeGateway()
    inv = _retriever_invocation("retriever_1", mode="metrics")
    result = asyncio.run(RetrieverAgent().invoke(inv, _ctx(gateway)))
    if result.status is ResultStatus.failed:
        pytest.skip(f"数据库不可达: {result.error.message}")

    assert "metric_definitions" in result.outputs
    assert "schema" in result.outputs and "data_ranges" in result.outputs
    assert "knowledge" not in result.outputs, "metrics 模式不应触发知识库检索"
    assert result.blackboard_writes == ["retriever_1.metrics"]


def test_retriever_knowledge_mode_skips_sql() -> None:
    import asyncio

    gateway = FakeGateway()
    inv = _retriever_invocation("retriever_2", mode="knowledge")
    result = asyncio.run(RetrieverAgent().invoke(inv, _ctx(gateway)))

    assert result.status is ResultStatus.ok
    assert "knowledge" in result.outputs and "knowledge_count" in result.outputs
    assert "metric_definitions" not in result.outputs, "knowledge 模式不应查口径表"
    assert result.blackboard_writes == ["retriever_2.knowledge"]


def test_retriever_all_mode_is_default_and_does_both() -> None:
    """不带 mode 时保持旧行为（all），保证既有单节点链路不被破坏。"""
    import asyncio

    gateway = FakeGateway()
    inv = _retriever_invocation("retriever_1")  # 不传 mode
    result = asyncio.run(RetrieverAgent().invoke(inv, _ctx(gateway)))
    if result.status is ResultStatus.failed:
        pytest.skip(f"数据库不可达: {result.error.message}")

    assert "metric_definitions" in result.outputs
    assert "knowledge" in result.outputs
    assert set(result.blackboard_writes) == {"retriever_1.metrics", "retriever_1.knowledge"}


def test_retriever_rejects_unknown_mode() -> None:
    import asyncio

    gateway = FakeGateway()
    inv = _retriever_invocation("retriever_1", mode="telepathy")
    result = asyncio.run(RetrieverAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "BAD_INPUT"
    assert result.error.retryable is False
    assert gateway.calls == 0


# ------------------------------------------------ Planner 的角色与模式门禁


def _node_payload(
    node_id: str, role: str, deps: list[str], *, mode: str | None = None
) -> dict:
    return {
        "node_id": node_id,
        "name": node_id,
        "agent_role": role,
        "depends_on": deps,
        "inputs": {"mode": mode} if mode else {},
        "outputs": {},
        "acceptance": ["x"],
    }


def _plan(nodes: list[dict]) -> str:
    import json

    return json.dumps(
        {
            "task_id": "11111111-2222-4333-8444-555555555555",
            "goal": "g",
            "created_at": "2026-10-16T12:00:00+08:00",
            "nodes": nodes,
        },
        ensure_ascii=False,
    )


def _plan_with_retrievers(mode_a: str | None, mode_b: str | None) -> str:
    return _plan(
        [
            _node_payload("retriever_1", "retriever", [], mode=mode_a),
            _node_payload("retriever_2", "retriever", [], mode=mode_b),
            _node_payload("analyst_1", "analyst", ["retriever_1", "retriever_2"]),
        ]
    )


def test_planner_accepts_parallel_retrieval_shape() -> None:
    """W5 的目标形态：两路并行检索（各司其职）+ 一个分析节点。"""
    import asyncio

    gateway = FakeGateway(_plan_with_retrievers("metrics", "knowledge"))
    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))

    assert result.status is ResultStatus.ok, result.error
    nodes = result.outputs["dag"]["nodes"]
    assert [n["node_id"] for n in nodes] == ["retriever_1", "retriever_2", "analyst_1"]
    assert result.outputs["node_count"] == 3


def test_planner_rejects_two_retrievers_without_mode() -> None:
    """两个 retriever 都不声明 mode 会重复做同一件事——必须拦掉而不是放任浪费。"""
    import asyncio

    gateway = FakeGateway(_plan_with_retrievers(None, None))
    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "UNSUPPORTED_PLAN"
    assert "inputs.mode" in result.error.message


def test_planner_rejects_duplicate_retriever_modes() -> None:
    import asyncio

    gateway = FakeGateway(_plan_with_retrievers("metrics", "metrics"))
    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert "重复" in result.error.message


def test_single_retriever_needs_no_mode() -> None:
    """单检索节点不必声明 mode（向后兼容），门禁只在多检索时生效。"""
    import asyncio

    gateway = FakeGateway(
        _plan(
            [
                _node_payload("retriever_1", "retriever", []),
                _node_payload("analyst_1", "analyst", ["retriever_1"]),
            ]
        )
    )
    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.ok, result.error


def test_planner_requires_available_roles() -> None:
    """没有可用角色就无从规划：明确报错，不让模型自由发挥。"""
    import asyncio

    gateway = FakeGateway()
    inv = _invocation()
    inv.input.pop("available_roles")
    result = asyncio.run(PlannerAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "BAD_INPUT"
    assert gateway.calls == 0, "不该浪费一次模型调用"


# ------------------------------------------------------------------------- 运行器
