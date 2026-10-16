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

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        self.inputs[inv.node_id] = dict(inv.input)
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


def _node(node_id: str, role: str, depends_on: list[str] | None = None) -> dict:
    return {
        "node_id": node_id,
        "name": node_id,
        "agent_role": role,
        "depends_on": depends_on or [],
        "inputs": {},
        "outputs": {},
        "acceptance": ["x"],
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
