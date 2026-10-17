"""动态 DAG 引擎测试。

**这个文件是 W5 的核心证据**：引擎的通用性不能靠生产链路证明（它只跑一种形状），
要靠这里用**合成 DAG** 证明——链、菱形、三路扇出、两层树都执行正确。

并行性用**可测量的方式**断言：假 Agent 带人工延迟并记录调用区间，
兄弟节点的时间区间必须真实重叠，且总耗时显著小于串行耗时。
"看起来是并行的"不算证据，时间区间重叠才算。

全部用例**不需要数据库、不需要模型**，因此快且确定。
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.agents import AgentContext, BaseAgent  # noqa: E402
from multiagent.agents.base import _Usage  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.contracts import (  # noqa: E402
    AgentInvocation,
    AgentResult,
    SourceRef,
    SourceType,
    SubtaskDAG,
)
from multiagent.gateway import LLMGateway, StubProvider  # noqa: E402
from multiagent.graph.pipeline import (  # noqa: E402
    AnalysisPipeline,
    PipelinePlanError,
    sink_node_ids,
)
from multiagent.tools import SqlSandbox  # noqa: E402

settings = Settings(_env_file=None)


class RecordingAgent(BaseAgent):
    """可配置延迟的假 Agent，记录每个节点的调用区间。

    同时记录收到的 input，用于验证「下游能拿到上游产物」。
    """

    def __init__(self, role: str, *, delay: float = 0.0) -> None:
        self.role = role
        self.version = "test"
        self._delay = delay
        self.intervals: dict[str, tuple[float, float]] = {}
        self.inputs: dict[str, dict] = {}
        self.visible_keys: dict[str, list[str]] = {}
        self.calls: dict[str, int] = {}

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        self.inputs[inv.node_id] = dict(inv.input)
        self.visible_keys[inv.node_id] = list(inv.visible_blackboard_keys)
        self.calls[inv.node_id] = self.calls.get(inv.node_id, 0) + 1
        await asyncio.sleep(self._delay)
        ended = time.perf_counter()
        self.intervals[inv.node_id] = (started, ended)
        return AgentResult(
            task_id=inv.task_id,
            node_id=inv.node_id,
            attempt=1,
            status="ok",
            tokens_used={"input": 1, "output": 1, "total": 2},
            duration_ms=int((ended - started) * 1000),
            outputs={"produced_by": inv.node_id, "summary": f"结论来自 {inv.node_id}"},
            sources=[SourceRef(type=SourceType.model_inference, ref="test")],
            blackboard_writes=[f"{inv.node_id}.out"],
        )


def _node(
    node_id: str,
    role: str,
    depends_on: list[str] | None = None,
    *,
    timeout_seconds: int = 300,
    max_retries: int = 2,
) -> dict:
    return {
        "node_id": node_id,
        "name": node_id,
        "agent_role": role,
        "depends_on": depends_on or [],
        "inputs": {},
        "outputs": {},
        "acceptance": ["x"],
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
    }


def _dag(nodes: list[dict]) -> dict:
    return {
        "task_id": "t-1",
        "goal": "g",
        "created_at": "2026-10-16T12:00:00+08:00",
        "nodes": nodes,
    }


def _ctx(agents: dict[str, BaseAgent]) -> AgentContext:
    return AgentContext(
        settings=settings,
        gateway=LLMGateway(primary=StubProvider()),
        sandbox=SqlSandbox(settings),
        registry=agents,
    )


async def _run(dag_json: dict, agents: dict[str, BaseAgent]) -> dict:
    """不接 checkpointer 直接跑图——引擎语义与持久化无关，这样测试更快更纯粹。"""
    pipeline = AnalysisPipeline(settings)
    graph = pipeline._build_from_dag(_ctx(agents), SubtaskDAG.model_validate(dag_json)).compile()
    return await graph.ainvoke(
        {
            "task_event": {"task_id": "t-1", "payload": {"text": "问题"}},
            "now": "2026-10-16T12:00:00+08:00",
            "dag": dag_json,
            "node_outputs": {},
            "blackboard": {},
            "node_results": [],
        }
    )


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _all_nodes(dag_json: dict) -> set[str]:
    return {n["node_id"] for n in dag_json["nodes"]}


# ------------------------------------------------------------------ 形状：链


def test_chain_executes_in_dependency_order() -> None:
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1"]),
            _node("reviewer_1", "reviewer", ["analyst_1"]),
        ]
    )
    agents = {r: RecordingAgent(r) for r in ("retriever", "analyst", "reviewer")}
    asyncio.run(_run(dag, agents))

    first = agents["retriever"].intervals["retriever_1"]
    second = agents["analyst"].intervals["analyst_1"]
    third = agents["reviewer"].intervals["reviewer_1"]
    assert first[1] <= second[0], "下游必须在依赖完成之后才开始"
    assert second[1] <= third[0]


# ------------------------------------------------------------ 形状：菱形并行


def test_diamond_runs_siblings_in_parallel_and_joins() -> None:
    """菱形是 W5 的目标形态：两个检索并行，分析等两者都完成。"""
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1", "retriever_2"]),
        ]
    )
    agents = {"retriever": RecordingAgent("retriever", delay=0.2), "analyst": RecordingAgent("analyst")}
    started = time.perf_counter()
    final = asyncio.run(_run(dag, agents))
    elapsed = time.perf_counter() - started

    left = agents["retriever"].intervals["retriever_1"]
    right = agents["retriever"].intervals["retriever_2"]
    join = agents["analyst"].intervals["analyst_1"]

    assert _overlap(left, right), f"两个检索节点没有并行: {left} vs {right}"
    assert max(left[1], right[1]) <= join[0], "分析必须等两个检索都完成"
    # 串行会是 0.4s+；并行应接近单次 0.2s
    assert elapsed < 0.35, f"总耗时 {elapsed:.2f}s 说明没有并行"
    assert set(final["node_results"][0]) >= {"node_id", "agent_role"}


def test_three_way_fanout_all_overlap() -> None:
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("retriever_3", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1", "retriever_2", "retriever_3"]),
        ]
    )
    agents = {"retriever": RecordingAgent("retriever", delay=0.2), "analyst": RecordingAgent("analyst")}
    started = time.perf_counter()
    asyncio.run(_run(dag, agents))
    elapsed = time.perf_counter() - started

    spans = [agents["retriever"].intervals[f"retriever_{i}"] for i in (1, 2, 3)]
    assert _overlap(spans[0], spans[1]) and _overlap(spans[1], spans[2])
    assert elapsed < 0.35, f"三路并行耗时 {elapsed:.2f}s 偏高"


def test_two_level_tree_respects_both_layers() -> None:
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1"]),
            _node("reviewer_1", "reviewer", ["retriever_2"]),
        ]
    )
    agents = {r: RecordingAgent(r, delay=0.15) for r in ("retriever", "analyst", "reviewer")}
    started = time.perf_counter()
    asyncio.run(_run(dag, agents))
    elapsed = time.perf_counter() - started

    # 第二层的两个节点也应并行 → 总耗时接近两跳，而不是四跳
    assert elapsed < 0.45, f"两层树耗时 {elapsed:.2f}s 说明第二层没有并行"


# ------------------------------------------------------ 并行下的状态不丢更新


def test_parallel_nodes_do_not_lose_blackboard_updates() -> None:
    """两个并行节点都写黑板时，两条都必须留下。

    状态字段没有 reducer（LastValue channel）时会直接抛 InvalidUpdateError；
    而「读旧状态再全量写回」会丢更新。这个用例同时守住这两点。
    """
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1", "retriever_2"]),
        ]
    )
    agents = {r: RecordingAgent(r, delay=0.1) for r in ("retriever", "analyst")}
    final = asyncio.run(_run(dag, agents))

    blackboard = final["blackboard"]
    assert {"retriever_1.out", "retriever_2.out"} <= set(blackboard), blackboard.keys()
    assert len(final["node_results"]) == 3, "三个节点的结果都要在"
    produced = {r["node_id"] for r in final["node_results"]}
    assert produced == _all_nodes(dag)


def test_parallel_node_outputs_are_both_visible_to_downstream() -> None:
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1", "retriever_2"]),
        ]
    )
    agents = {r: RecordingAgent(r, delay=0.05) for r in ("retriever", "analyst")}
    asyncio.run(_run(dag, agents))

    received = agents["analyst"].inputs["analyst_1"]
    # 上游产物按 key 合并后交给下游：两路的 produced_by 都能看到（后者覆盖前者）
    assert "produced_by" in received
    assert received["upstream_nodes"] == ["retriever_1", "retriever_2"]
    assert received["question"] == "问题"
    assert received["acceptance"] == ["x"]


def test_node_declared_inputs_are_passed_through() -> None:
    """节点自带的 inputs（如 retriever 的 mode）要出现在 invocation 里。"""
    dag = _dag(
        [
            {**_node("retriever_2", "retriever"), "inputs": {"mode": "knowledge"}},
            _node("analyst_1", "analyst", ["retriever_2"]),
        ]
    )
    agents = {"retriever": RecordingAgent("retriever"), "analyst": RecordingAgent("analyst")}
    asyncio.run(_run(dag, agents))
    assert agents["retriever"].inputs["retriever_2"]["mode"] == "knowledge"


# ------------------------------------------------------------------ 失败语义


def test_unregistered_role_fails_before_running_anything() -> None:
    """未注册角色必须在**建图时**就报错，不能跑到那个节点才发现。

    早失败比晚失败好：等到那时前序节点的 token 已经烧掉了。
    """
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("reviewer_1", "reviewer", ["retriever_1"]),
        ]
    )
    agents = {"retriever": RecordingAgent("retriever")}  # 没有 reviewer
    with pytest.raises(PipelinePlanError, match="reviewer_1"):
        asyncio.run(_run(dag, agents))


def test_unregistered_role_message_lists_available_roles() -> None:
    dag = _dag([_node("writer_1", "writer")])
    with pytest.raises(PipelinePlanError) as excinfo:
        asyncio.run(_run(dag, {"retriever": RecordingAgent("retriever")}))
    assert "retriever" in str(excinfo.value), "报错要告诉使用者有哪些角色可用"


# ------------------------------------------------------------------ 汇点判定


def test_sink_node_ids_finds_terminal_nodes() -> None:
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1", "retriever_2"]),
        ]
    )
    assert sink_node_ids(dag) == ["analyst_1"]


def test_sink_node_ids_handles_multiple_sinks() -> None:
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1"]),
            _node("reviewer_1", "reviewer", ["retriever_1"]),
        ]
    )
    assert set(sink_node_ids(dag)) == {"analyst_1", "reviewer_1"}


def test_multi_sink_graph_executes_both_branches() -> None:
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1"]),
            _node("reviewer_1", "reviewer", ["retriever_1"]),
        ]
    )
    agents = {r: RecordingAgent(r) for r in ("retriever", "analyst", "reviewer")}
    final = asyncio.run(_run(dag, agents))
    assert {r["node_id"] for r in final["node_results"]} == _all_nodes(dag)


# ------------------------------------------------------------------ 黑板订阅


def test_join_waits_for_all_dependencies_and_runs_once() -> None:
    """依赖横跨两层时，下游必须等**全部**上游完成，且只执行一次。

    回归 W6 抓到的引擎缺陷：逐个 add_edge(dep, node) 在 LangGraph 里是
    「任一上游完成即触发」——reviewer_1 曾在 analyst_1 完成前就提前跑了一次，
    之后又跑第二次。修复是改用列表形式连边（join 语义）。
    """
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1"]),
            _node("reviewer_1", "reviewer", ["retriever_1", "retriever_2", "analyst_1"]),
        ]
    )
    agents = {r: RecordingAgent(r) for r in ("retriever", "analyst", "reviewer")}
    final = asyncio.run(_run(dag, agents))

    assert agents["reviewer"].calls["reviewer_1"] == 1, "下游不得提前执行或重复执行"
    # 唯一一次执行时，三个上游的条目全部可见
    assert set(agents["reviewer"].inputs["reviewer_1"]["blackboard"]) == {
        "retriever_1.out",
        "retriever_2.out",
        "analyst_1.out",
    }
    # node_results 里也不得出现重复节点（重复即意味着跑了不止一次）
    ids = [r["node_id"] for r in final["node_results"]]
    assert len(ids) == len(set(ids))


def test_blackboard_entries_are_contract_conformant_in_state() -> None:
    """图跑完后，state 里的黑板条目必须能被冻结契约解析（并行路径也一样）。"""
    from multiagent.contracts import BlackboardEntry

    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1", "retriever_2"]),
        ]
    )
    agents = {"retriever": RecordingAgent("retriever"), "analyst": RecordingAgent("analyst")}
    final = asyncio.run(_run(dag, agents))

    for key, entry in final["blackboard"].items():
        parsed = BlackboardEntry.model_validate(entry)
        assert parsed.key == key
        assert parsed.version == 1
        assert parsed.written_by.node_id == key.split(".", 1)[0]


def test_downstream_sees_upstream_blackboard_and_visible_keys() -> None:
    """订阅语义：下游只拿到**直接上游**的条目，且可见性字段被真实填充。"""
    dag = _dag(
        [
            _node("retriever_1", "retriever"),
            _node("retriever_2", "retriever"),
            _node("analyst_1", "analyst", ["retriever_1"]),
            _node("reviewer_1", "reviewer", ["retriever_1", "retriever_2", "analyst_1"]),
        ]
    )
    agents = {
        r: RecordingAgent(r) for r in ("retriever", "analyst", "reviewer")
    }
    asyncio.run(_run(dag, agents))

    # analyst 只依赖 retriever_1：看不到 retriever_2 的条目
    analyst_input = agents["analyst"].inputs["analyst_1"]
    assert set(analyst_input["blackboard"]) == {"retriever_1.out"}
    assert agents["analyst"].visible_keys["analyst_1"] == ["retriever_1.out"]

    # reviewer 依赖全部三个：三条条目都可见，且能看到来源明细
    reviewer_input = agents["reviewer"].inputs["reviewer_1"]
    assert set(reviewer_input["blackboard"]) == {
        "retriever_1.out",
        "retriever_2.out",
        "analyst_1.out",
    }
    assert agents["reviewer"].visible_keys["reviewer_1"] == sorted(reviewer_input["blackboard"])
    assert reviewer_input["upstream_sources"]["analyst_1"][0]["type"] == "model_inference"


# --------------------------------------------- W7：四大约束熔断（确定性证据）


def _run_with_budget(dag_json: dict, agents: dict, budget: dict | None) -> dict:
    """用带预算的初始状态跑图（预算走 task_event 的序列化形态）。"""
    from datetime import datetime, timedelta, timezone

    task_event = {"task_id": "t-1", "payload": {"text": "问题"}}
    if budget is not None:
        task_event["budget"] = {
            "max_steps": budget.get("max_steps", 30),
            "max_tokens": budget.get("max_tokens", 200_000),
            "max_retries": budget.get("max_retries", 2),
            "deadline": budget.get(
                "deadline",
                (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            ),
        }
    pipeline = AnalysisPipeline(settings)
    graph = pipeline._build_from_dag(
        _ctx(agents), SubtaskDAG.model_validate(dag_json)
    ).compile()
    return asyncio.run(
        graph.ainvoke(
            {
                "task_event": task_event,
                "now": "2026-10-17T12:00:00+08:00",
                "dag": dag_json,
                "node_outputs": {},
                "blackboard": {},
                "node_results": [],
                "steps_used": budget.get("steps_used", 0) if budget else 0,
                "tokens_used": budget.get("tokens_used", 0) if budget else 0,
            }
        )
    )


def test_breaker_stops_when_step_limit_reached() -> None:
    """步数超限：后续节点**不启动**，以 STEP_LIMIT 熔断。"""
    from multiagent.graph.pipeline import PipelineNodeFailed

    dag = _dag([_node("retriever_1", "retriever"), _node("analyst_1", "analyst", ["retriever_1"])])
    agents = {"retriever": RecordingAgent("retriever"), "analyst": RecordingAgent("analyst")}

    with pytest.raises(PipelineNodeFailed) as excinfo:
        _run_with_budget(dag, agents, {"max_steps": 1, "steps_used": 0})
    assert excinfo.value.code == "STEP_LIMIT"
    assert excinfo.value.node_id == "analyst_1", "第二个节点启动前就该被拦下"
    assert agents["analyst"].calls == {}, "超限节点不得执行"


def test_breaker_stops_when_token_budget_exhausted() -> None:
    """Token 超限：以 BUDGET_EXCEEDED 熔断（按已消耗累计判断）。"""
    from multiagent.graph.pipeline import PipelineNodeFailed

    dag = _dag([_node("retriever_1", "retriever"), _node("analyst_1", "analyst", ["retriever_1"])])
    agents = {"retriever": RecordingAgent("retriever"), "analyst": RecordingAgent("analyst")}

    with pytest.raises(PipelineNodeFailed) as excinfo:
        _run_with_budget(
            dag, agents, {"max_tokens": 100, "tokens_used": 150}
        )
    assert excinfo.value.code == "BUDGET_EXCEEDED"
    assert excinfo.value.node_id == "retriever_1", "第一个节点启动前就应熔断"


def test_breaker_stops_when_deadline_passed() -> None:
    from multiagent.graph.pipeline import PipelineNodeFailed

    dag = _dag([_node("retriever_1", "retriever")])
    agents = {"retriever": RecordingAgent("retriever")}

    with pytest.raises(PipelineNodeFailed) as excinfo:
        _run_with_budget(
            dag, agents, {"deadline": "2026-01-01T00:00:00+08:00"}
        )
    assert excinfo.value.code == "DEADLINE_EXCEEDED"
    assert agents["retriever"].calls == {}


def test_normal_budget_does_not_trip() -> None:
    dag = _dag([_node("retriever_1", "retriever"), _node("analyst_1", "analyst", ["retriever_1"])])
    agents = {"retriever": RecordingAgent("retriever"), "analyst": RecordingAgent("analyst")}
    final = _run_with_budget(dag, agents, {"max_steps": 30, "max_tokens": 60_000})
    assert {r["node_id"] for r in final["node_results"]} == {"retriever_1", "analyst_1"}
    # 计数随图累积（Planner 的 1 步在真实链路里由 _plan 预置）
    assert final["steps_used"] == 2
    assert final["tokens_used"] == 4  # 每节点 tokens_used.total=2


# --------------------------------------------- W7：失败路径保真


def test_failed_outcome_keeps_partial_results_and_tokens() -> None:
    """失败不得再「零 token、零节点」：熔断计数与成本归因都依赖这份保真。"""
    import time
    from types import SimpleNamespace

    from multiagent.graph.pipeline import AnalysisPipeline, PipelineNodeFailed

    exc = PipelineNodeFailed("analyst_1", "PROVIDER_UNAVAILABLE", "上游超时", True)
    dag = _dag([_node("retriever_1", "retriever"), _node("analyst_1", "analyst", ["retriever_1"])])
    state = {
        "dag": dag,
        "node_results": [
            {
                "node_id": "retriever_1",
                "agent_role": "retriever",
                "status": "ok",
                "tokens_used": {"input": 10, "output": 13, "total": 123},
                "outputs": {"metric_count": 6, "summary": "部分结论"},
            }
        ],
        "node_outputs": {"retriever_1": {"metric_count": 6, "summary": "部分结论"}},
        "blackboard": {"retriever_1.metrics": {"version": 1, "key": "retriever_1.metrics"}},
    }
    outcome = AnalysisPipeline(settings)._failed_outcome(
        "t-1", exc, state, SimpleNamespace(history=[]), time.perf_counter()
    )

    assert outcome.status == "failed"
    assert outcome.total_tokens == 123, "已消耗 token 必须保留"
    assert outcome.node_results, "已完成节点必须保留"
    assert outcome.blackboard, "黑板条目必须保留"
    assert outcome.error_code == "PROVIDER_UNAVAILABLE"
    assert outcome.retryable is True
    assert "节点 analyst_1" in outcome.error
    assert outcome.summary == "部分结论", "部分结论可用于观测"


def test_read_checkpoint_state_is_best_effort() -> None:
    """读 checkpoint 失败不得掩盖原始错误：返回空 dict。"""
    from multiagent.graph.pipeline import AnalysisPipeline

    class _Saver:
        def __init__(self, snapshot=None, boom: bool = False) -> None:
            self._snapshot = snapshot
            self._boom = boom

        async def aget(self, config):
            if self._boom:
                raise RuntimeError("checkpoint 读不了")
            return self._snapshot

    class _Snap:
        def __init__(self, values):
            self.checkpoint = {"channel_values": values}

    async def run(saver):
        return await AnalysisPipeline._read_checkpoint_state(saver, {})

    assert asyncio.run(run(_Saver(None))) == {}
    assert asyncio.run(run(_Saver(boom=True))) == {}
    assert asyncio.run(run(_Saver(_Snap({"node_results": [1]})))) == {"node_results": [1]}


def test_new_task_event_budget_comes_from_settings() -> None:
    """任务预算从 Settings 派生（D2=60,000），而不是契约默认的 200,000。"""
    from datetime import datetime, timedelta, timezone

    from multiagent.config import Settings
    from multiagent.graph.pipeline import new_task_event

    cfg = Settings(
        _env_file=None,
        default_max_tokens=1234,
        default_max_steps=7,
        default_max_retries=1,
    )
    event = new_task_event("q", settings=cfg)

    assert event.budget.max_tokens == 1234
    assert event.budget.max_steps == 7
    assert event.budget.max_retries == 1
    assert event.budget.deadline > datetime.now(timezone.utc) + timedelta(minutes=4)


# ------------------------------------------------- W7：节点级重试与超时


class FlakyAgent(BaseAgent):
    """按脚本失败/成功的假 Agent：记录每次 attempt，便于断言重试语义。"""

    def __init__(
        self,
        role: str,
        *,
        failures_left: int = 0,
        retryable: bool = True,
        delay: float = 0.0,
    ) -> None:
        self.role = role
        self.version = "test"
        self.calls: dict[str, int] = {}
        self.attempts: list[int] = []
        self._failures_left = failures_left
        self._retryable = retryable
        self._delay = delay

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        self.attempts.append(inv.attempt)
        self.calls[inv.node_id] = self.calls.get(inv.node_id, 0) + 1
        await asyncio.sleep(self._delay)
        if self._failures_left > 0:
            self._failures_left -= 1
            return AgentResult(
                task_id=inv.task_id,
                node_id=inv.node_id,
                attempt=inv.attempt,
                status="failed",
                tokens_used={"input": 5, "output": 5, "total": 10},
                duration_ms=1,
                error={
                    "code": "PROVIDER_UNAVAILABLE",
                    "message": "模拟供应商抖动",
                    "retryable": self._retryable,
                },
            )
        return AgentResult(
            task_id=inv.task_id,
            node_id=inv.node_id,
            attempt=inv.attempt,
            status="ok",
            tokens_used={"input": 1, "output": 1, "total": 2},
            duration_ms=1,
            outputs={"produced_by": inv.node_id, "summary": f"{inv.node_id} 完成"},
            sources=[SourceRef(type=SourceType.model_inference, ref="test")],
            blackboard_writes=[f"{inv.node_id}.out"],
        )


def test_retryable_failure_is_retried_then_succeeds() -> None:
    """可重试失败 → attempt 递增重试；node_results 只保留最终结果（一份）。"""
    dag = _dag([_node("retriever_1", "retriever")])
    agent = FlakyAgent("retriever", failures_left=1)
    final = _run_with_budget(dag, {"retriever": agent}, {"max_steps": 30, "max_tokens": 60_000})

    assert agent.attempts == [1, 2], "应恰好重试一次"
    results = [r for r in final["node_results"] if r["node_id"] == "retriever_1"]
    assert len(results) == 1, "重试成功后不得重复写 node_results"
    assert results[0]["attempt"] == 2, "最终结果应记录成功的 attempt"


def test_retry_exhausted_raises_last_error() -> None:
    """重试耗尽：抛最后一次的错误，attempt 用满 1+min(节点,任务)。"""
    from multiagent.graph.pipeline import PipelineNodeFailed

    dag = _dag([_node("retriever_1", "retriever", max_retries=2)])
    agent = FlakyAgent("retriever", failures_left=10)

    with pytest.raises(PipelineNodeFailed) as excinfo:
        _run_with_budget(
            dag, {"retriever": agent}, {"max_steps": 30, "max_tokens": 60_000, "max_retries": 2}
        )
    assert excinfo.value.code == "PROVIDER_UNAVAILABLE"
    assert agent.attempts == [1, 2, 3], "1 次初始 + 2 次重试"


def test_task_level_max_retries_caps_node_retries() -> None:
    """任务预算的 max_retries 收紧时，节点重试随之收紧（取两者较小值）。"""
    from multiagent.graph.pipeline import PipelineNodeFailed

    dag = _dag([_node("retriever_1", "retriever", max_retries=5)])
    agent = FlakyAgent("retriever", failures_left=10)

    with pytest.raises(PipelineNodeFailed):
        _run_with_budget(
            dag, {"retriever": agent}, {"max_steps": 30, "max_tokens": 60_000, "max_retries": 0}
        )
    assert agent.attempts == [1], "任务预算 max_retries=0 时不得重试"


def test_non_retryable_failure_is_not_retried() -> None:
    from multiagent.graph.pipeline import PipelineNodeFailed

    dag = _dag([_node("retriever_1", "retriever", max_retries=2)])
    agent = FlakyAgent("retriever", failures_left=10, retryable=False)

    with pytest.raises(PipelineNodeFailed) as excinfo:
        _run_with_budget(dag, {"retriever": agent}, {"max_steps": 30, "max_tokens": 60_000})
    assert excinfo.value.retryable is False
    assert agent.attempts == [1], "不可重试错误不得重试"


def test_node_timeout_raises_timeout_code() -> None:
    """节点超时：按 TIMEOUT 失败（可重试，但本用例 max_retries=0）。"""
    from multiagent.graph.pipeline import PipelineNodeFailed

    dag = _dag([_node("retriever_1", "retriever", timeout_seconds=1, max_retries=0)])
    agent = FlakyAgent("retriever", delay=1.3)

    with pytest.raises(PipelineNodeFailed) as excinfo:
        _run_with_budget(dag, {"retriever": agent}, {"max_steps": 30, "max_tokens": 60_000})
    assert excinfo.value.code == "TIMEOUT"
    assert excinfo.value.retryable is True


# ------------------------------------------------- W7：HITL 审批门（interrupt + 恢复）


def _run_hitl(
    dag_json: dict,
    agents: dict,
    *,
    decisions: list[dict] | None = None,
    budget: dict | None = None,
):
    """编译带 checkpointer 的图：首次 invoke 会暂停在审批门，再按 decisions 逐次恢复。"""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from datetime import datetime, timedelta, timezone

    pipeline = AnalysisPipeline(settings)
    saver = MemorySaver()
    graph = pipeline._build_from_dag(
        _ctx(agents), SubtaskDAG.model_validate(dag_json)
    ).compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "t-hitl"}}
    task_event = {"task_id": "t-hitl", "payload": {"text": "写一份对外公告并发送给客户"}}
    if budget is not None:
        task_event["budget"] = {
            "max_steps": budget.get("max_steps", 30),
            "max_tokens": budget.get("max_tokens", 200_000),
            "max_retries": budget.get("max_retries", 2),
            "deadline": budget.get(
                "deadline",
                (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            ),
        }
    state = {
        "task_event": task_event,
        "now": "2026-10-17T12:00:00+08:00",
        "dag": dag_json,
        "node_outputs": {},
        "blackboard": {},
        "node_results": [],
        "steps_used": 0,
        "tokens_used": 0,
    }
    snapshots = [asyncio.run(graph.ainvoke(state, config=config))]
    for decision in decisions or []:
        snapshots.append(
            asyncio.run(graph.ainvoke(Command(resume=decision), config=config))
        )
    return snapshots, graph, config


def _hitl_dag() -> dict:
    # analyst_1 需要人工审批；writer_1 依赖 analyst_1（必须等审批通过才放行）
    return _dag(
        [
            _node("retriever_1", "retriever"),
            {
                **_node("analyst_1", "analyst", ["retriever_1"]),
                "requires_human_approval": True,
            },
            _node("writer_1", "writer", ["analyst_1"]),
        ]
    )


def _hitl_agents() -> dict:
    return {
        "retriever": RecordingAgent("retriever"),
        "analyst": RecordingAgent("analyst"),
        "writer": RecordingAgent("writer"),
    }


def test_approval_gate_pauses_with_payload_and_blocks_downstream() -> None:
    agents = _hitl_agents()
    snapshots, _, _ = _run_hitl(_hitl_dag(), agents)

    first = snapshots[0]
    assert first.get("__interrupt__"), "首次运行必须暂停在审批门"
    payload = list(first["__interrupt__"])[0].value
    assert payload["kind"] == "node_approval"
    assert payload["node_id"] == "analyst_1"
    assert payload["outputs"]["produced_by"] == "analyst_1", "审批载荷要带节点产物"
    assert "analyst_1.out" in payload["blackboard_keys"]
    # 下游不得提前执行；被审节点本身只执行一次（恢复时重放的是门，不是 Agent）
    assert agents["analyst"].calls == {"analyst_1": 1}
    assert agents["writer"].calls == {}, "审批未决时下游不得执行"


def test_approval_resume_approved_releases_downstream() -> None:
    agents = _hitl_agents()
    snapshots, _, _ = _run_hitl(
        _hitl_dag(), agents, decisions=[{"decision": "approved", "comment": "同意"}]
    )

    final = snapshots[-1]
    assert not final.get("__interrupt__"), "批准后不应再暂停"
    assert agents["analyst"].calls == {"analyst_1": 1}, "恢复只重放审批门，不重放 Agent"
    assert agents["writer"].calls == {"writer_1": 1}, "批准后下游应执行"
    assert final["approvals"]["analyst_1"]["decision"] == "approved"


def test_approval_rejected_stops_without_running_downstream() -> None:
    """驳回：下游节点在执行前短路（APPROVAL_REJECTED），Agent 不运行。

    不用 Command(goto=END)——实测它不会压制静态边，下游照样会跑。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from multiagent.graph.pipeline import AnalysisPipeline, PipelineNodeFailed

    agents = _hitl_agents()
    pipeline = AnalysisPipeline(settings)
    graph = pipeline._build_from_dag(
        _ctx(agents), SubtaskDAG.model_validate(_hitl_dag())
    ).compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t-rejected"}}
    state = {
        "task_event": {"task_id": "t-rejected", "payload": {"text": "对外公告"}},
        "now": "2026-10-17T12:00:00+08:00",
        "dag": _hitl_dag(),
        "node_outputs": {},
        "blackboard": {},
        "node_results": [],
        "steps_used": 0,
        "tokens_used": 0,
    }
    first = asyncio.run(graph.ainvoke(state, config=config))
    assert first.get("__interrupt__")

    with pytest.raises(PipelineNodeFailed) as excinfo:
        asyncio.run(
            graph.ainvoke(Command(resume={"decision": "rejected"}), config=config)
        )
    assert excinfo.value.code == "APPROVAL_REJECTED"
    assert agents["writer"].calls == {}, "驳回后下游不得执行"
    snap = asyncio.run(graph.aget_state(config))
    assert snap.values["approvals"]["analyst_1"]["decision"] == "rejected"


def test_approval_rewritten_replaces_outputs_and_blackboard() -> None:
    agents = _hitl_agents()
    snapshots, _, _ = _run_hitl(
        _hitl_dag(),
        agents,
        decisions=[
            {
                "decision": "rewritten",
                "comment": "人工修订",
                "outputs": {"produced_by": "analyst_1", "summary": "人工改写后的结论"},
            }
        ],
    )

    final = snapshots[-1]
    assert agents["writer"].calls == {"writer_1": 1}
    # 下游拿到的是人工改写后的产物（gate 覆盖 node_outputs）
    assert final["node_outputs"]["analyst_1"]["summary"] == "人工改写后的结论"
    assert (
        final["node_outputs"]["writer_1"]["produced_by"] == "writer_1"
    ), "writer 应已成功执行"
    entry = final["blackboard"]["analyst_1.out"]
    assert entry["version"] == 2, "人工改写以版本 +1 写入黑板"
    assert entry["status"] == "validated"
    assert entry["payload"]["summary"] == "人工改写后的结论"


def test_approval_gate_not_inserted_when_not_required() -> None:
    """没有 requires_human_approval 的 DAG 不得多出审批门（既有链路零变化）。"""
    dag = _dag(
        [_node("retriever_1", "retriever"), _node("analyst_1", "analyst", ["retriever_1"])]
    )
    agents = {"retriever": RecordingAgent("retriever"), "analyst": RecordingAgent("analyst")}
    snapshots, _, _ = _run_hitl(dag, agents)
    assert not snapshots[0].get("__interrupt__")
    assert agents["analyst"].calls == {"analyst_1": 1}
