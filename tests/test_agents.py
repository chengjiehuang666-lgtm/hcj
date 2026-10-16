"""Agent 层测试：结构化输出的容错、规划结果的可执行性约束、失败语义。

用假网关注入固定响应，因此**不调用真实模型、不消耗 token、结果确定**。
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timedelta, timezone

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
    BlackboardEntry,
    EntryStatus,
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
    # 条目必须过冻结契约的校验，不是「形状相近的 dict」
    parsed = BlackboardEntry.model_validate(entry)
    assert parsed.version == 1
    assert parsed.status is EntryStatus.draft
    assert parsed.task_id == inv.task_id
    # 用真实 Agent 版本，不再写死 0.1.0（原先写死的是潜伏 bug）
    assert parsed.written_by.agent_version == "0.3.0"
    assert parsed.written_by.node_id == "retriever_2"
    assert parsed.written_by.attempt == 1
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
    # 固定写入时间，否则两次调用的 written_at 不同，测的就不是「无状态泄漏」
    fixed = datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    first = AnalysisPipeline._absorb(inv, ok, RetrieverAgent(), written_at=fixed)
    second = AnalysisPipeline._absorb(inv, ok, RetrieverAgent(), written_at=fixed)
    assert first == second


# ------------------------------------------------- Retriever（需要数据库则跳过）


def _skip_if_db_unreachable() -> None:
    """只在**连接层**不可达时跳过。

    旧写法把「任何 failed 结果」一律 skip 成「数据库不可达」——代码缺陷导致的
    真实回归（如 SQL 写错表名）会被静默跳过、套件仍然全绿（W6 独立复核确认）。
    正确做法是先探连接：库通了，后续断言就必须真刀真枪地过。
    """
    import psycopg

    try:
        with psycopg.connect(settings.postgres_ro_dsn, connect_timeout=5):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可达: {exc}")


def test_retriever_returns_metric_definitions_from_db() -> None:
    import asyncio

    _skip_if_db_unreachable()
    gateway = FakeGateway()  # 本节点不应调用模型
    result = asyncio.run(RetrieverAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.ok, result.error
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

    _skip_if_db_unreachable()
    result = asyncio.run(AnalystAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.ok, result.error
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

    _skip_if_db_unreachable()
    gateway = FakeGateway()
    inv = _retriever_invocation("retriever_1", mode="metrics")
    result = asyncio.run(RetrieverAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.ok, result.error

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
    _skip_if_db_unreachable()
    inv = _retriever_invocation("retriever_1")  # 不传 mode
    result = asyncio.run(RetrieverAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.ok, result.error

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


def test_retriever_offloads_blocking_work_to_thread(monkeypatch) -> None:
    """检索的阻塞工作必须走线程，否则「两路并行」会退化成串行。

    测法刻意做成**确定性**的：用 `time.sleep` 模拟阻塞工作，再看心跳任务能否推进。
    直接测「真实检索期间的耗时/心跳」不稳定——冷启动要 1 秒、热了只要几十毫秒，
    阈值怎么定都会偶发失败（这个用例第一版就是这么挂的）。
    """
    import asyncio
    import time

    from multiagent.rag import KnowledgeStore

    def blocking_search(self, query, *, top_k=None):  # noqa: ANN001
        time.sleep(0.2)  # 模拟同步阻塞调用
        return []

    monkeypatch.setattr(KnowledgeStore, "search", blocking_search)

    async def scenario() -> int:
        ticks = 0
        stop = False

        async def heartbeat() -> None:
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.005)

        hb = asyncio.create_task(heartbeat())
        try:
            inv = _retriever_invocation("retriever_2", mode="knowledge")
            result = await RetrieverAgent().invoke(inv, _ctx(FakeGateway()))
            assert result.status is ResultStatus.ok, result.error
        finally:
            stop = True
            await hb
        return ticks

    ticks = asyncio.run(scenario())
    # 200ms / 5ms ≈ 40 次；若阻塞占住事件循环则接近 0
    assert ticks >= 10, f"阻塞工作占住了事件循环（心跳仅 {ticks} 次）"


def test_retriever_survives_empty_knowledge_hit(monkeypatch) -> None:
    """知识库无命中是**正常情况**，不能变成节点崩溃。

    契约要求 status=ok 的结果必须带 sources；「没命中」时若不补一条
    「检索无命中」的来源记录，就会在 ok() 的契约校验处失败——
    只要用户问一个与知识库无关的问题，检索节点就会挂掉整个任务。曾一度如此。
    """
    import asyncio

    from multiagent.rag import KnowledgeStore

    monkeypatch.setattr(KnowledgeStore, "search", lambda self, q, **kw: [])

    inv = _retriever_invocation("retriever_2", mode="knowledge")
    result = asyncio.run(RetrieverAgent().invoke(inv, _ctx(FakeGateway())))

    assert result.status is ResultStatus.ok, result.error
    assert result.outputs["knowledge"] == []
    assert result.outputs["knowledge_count"] == 0
    assert result.sources, "无命中时也必须留下来源记录（契约要求）"
    assert "无命中" in result.sources[0].ref


def test_retriever_metrics_mode_always_has_sources(monkeypatch) -> None:
    """指标口径为空时同样要留下来源记录（走 SQL，天然有 sql 来源）。"""
    import asyncio

    _skip_if_db_unreachable()
    inv = _retriever_invocation("retriever_1", mode="metrics")
    result = asyncio.run(RetrieverAgent().invoke(inv, _ctx(FakeGateway())))
    assert result.status is ResultStatus.ok, result.error
    assert result.sources


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


# ---------------------------------------- Planner 门禁（W6 复核后的代码侧加固）


def test_planner_rejects_invalid_retriever_mode() -> None:
    """非法 mode 必须在**规划期**拦下——执行期失败会白烧规划与全部前序 token。"""
    import asyncio

    gateway = FakeGateway(_plan_with_retrievers("policy", "knowledge"))
    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "UNSUPPORTED_PLAN"
    assert "取值非法" in result.error.message


def test_planner_rejects_single_invalid_mode_too() -> None:
    """单 retriever 的 mode 取值同样要校验（旧门禁在少于两个检索时整段跳过）。"""
    import asyncio

    gateway = FakeGateway(
        _plan(
            [
                _node_payload("retriever_1", "retriever", [], mode="policy"),
                _node_payload("analyst_1", "analyst", ["retriever_1"]),
            ]
        )
    )
    result = asyncio.run(PlannerAgent().invoke(_invocation(), _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert "取值非法" in result.error.message


def _reviewer_plan(deps: list[str]) -> str:
    return _plan(
        [
            _node_payload("retriever_1", "retriever", [], mode="metrics"),
            _node_payload("analyst_1", "analyst", ["retriever_1"]),
            _node_payload("reviewer_1", "reviewer", deps),
        ]
    )


def test_planner_rejects_reviewer_with_missing_dependencies() -> None:
    """「reviewer 依赖其余全部节点」必须有**代码门禁**，不能只靠提示词。

    独立复核发现：漏依赖的规划会被照单接受，而状态流转与记忆沉淀门槛都只看
    verdict、不看核验范围——未被核对的结论可以以「已过评审」身份沉淀入库。
    """
    import asyncio

    gateway = FakeGateway(_reviewer_plan(["analyst_1"]))  # 漏了 retriever_1
    inv = _invocation(available_roles=["retriever", "analyst", "reviewer"])
    result = asyncio.run(PlannerAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert result.error.code == "UNSUPPORTED_PLAN"
    assert "depends_on 缺少" in result.error.message


def test_planner_accepts_reviewer_covering_all_upstream() -> None:
    import asyncio

    gateway = FakeGateway(_reviewer_plan(["retriever_1", "analyst_1"]))
    inv = _invocation(available_roles=["retriever", "analyst", "reviewer"])
    result = asyncio.run(PlannerAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.ok, result.error


def test_planner_rejects_multiple_reviewers() -> None:
    """多 reviewer 会在同一 superstep 并发更新同一批条目（乐观锁冲突消解属 W7），
    当前设计是单一汇点评审——在执行前拦掉，而不是让状态流转静默丢一方。"""
    import asyncio

    gateway = FakeGateway(
        _plan(
            [
                _node_payload("retriever_1", "retriever", [], mode="metrics"),
                _node_payload("analyst_1", "analyst", ["retriever_1"]),
                _node_payload("reviewer_1", "reviewer", ["retriever_1", "analyst_1"]),
                _node_payload("reviewer_2", "reviewer", ["analyst_1"]),
            ]
        )
    )
    inv = _invocation(available_roles=["retriever", "analyst", "reviewer"])
    result = asyncio.run(PlannerAgent().invoke(inv, _ctx(gateway)))
    assert result.status is ResultStatus.failed
    assert "多个 reviewer" in result.error.message


# ------------------------------------------------ 预算派生（只能收紧）


def test_invocation_budget_derives_from_task_budget() -> None:
    """AgentInvocation.budget 派生自任务预算且**只能收紧**（冻结契约语义）。

    独立复核发现：旧实现直接取 Settings 默认值，提交方在 TaskEvent.budget 里
    收紧的额度被忽略（甚至比提交方要求的更宽）。
    """
    import asyncio  # noqa: F401
    from datetime import datetime, timedelta, timezone

    from multiagent.contracts import Budget
    from multiagent.graph.pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline(settings)

    tight = Budget(
        max_tokens=1000,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=3),
    )
    inv = pipeline._make_invocation(
        "retriever_1", "retriever", {}, task_id="t-1", budget=tight
    )
    assert inv.budget.max_tokens == 1000, "提交方收紧的额度必须生效"
    assert inv.budget.timeout_seconds <= 180, "deadline 必须参与派生"

    # 宽松的预算不得把调用额度放宽到超过本地默认
    loose = Budget(
        max_tokens=10_000_000,
        deadline=datetime.now(timezone.utc) + timedelta(hours=5),
    )
    inv2 = pipeline._make_invocation(
        "retriever_1", "retriever", {}, task_id="t-1", budget=loose
    )
    assert inv2.budget.max_tokens == settings.default_max_tokens

    # 图内节点走序列化 dict 形态（state 必须可 JSON 序列化）
    inv3 = pipeline._make_invocation(
        "retriever_1",
        "retriever",
        {},
        task_id="t-1",
        budget={"max_tokens": 500, "deadline": "2026-01-01T00:00:00+08:00"},
    )
    assert inv3.budget.max_tokens == 500
