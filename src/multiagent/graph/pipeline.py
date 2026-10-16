"""编排层：由 SubtaskDAG 在运行时驱动的动态图执行。

**W5 的核心变化**：链路不再写死，而是由 Planner 产出的 DAG 决定图结构：
节点 = DAG 节点，边 = depends_on，同一层互不依赖的节点由 LangGraph 在同一个
superstep 内**并发执行**。引擎是通用的（能跑任意合法 DAG），"产出什么形状"是
Planner 的策略，两者分开——引擎的通用性由 `tests/test_pipeline_engine.py` 用多种
合成 DAG 证明。

**为什么 Planner 在 graph 之外先跑**：图结构依赖它的输出，无法装进同一个图；
契约本身也写明「Planner 是产出者而非节点」（`contracts/subtask_dag.py` 的
`AgentRole` 注释）。

**三个并发相关的硬约束**（W5 改造时必须同时解决，否则并行节点会互相破坏）：

1. `node_outputs` / `blackboard` / `node_results` 必须带 reducer。
   无 reducer 的字段是 `LastValue` channel，同一步内被两个节点写入会直接抛
   `InvalidUpdateError`（`langgraph/channels/last_value.py`）。
2. 节点函数只能返回**本节点的增量**，不能读旧状态再全量写回——并行时那会丢更新
   （各自的读都基于步初快照）。
3. 状态里不再有按 role 硬编码的扁平字段（`schema`/`knowledge`/`summary`…）。
   那套分发是「role 三选一」，新角色的输出会被静默丢弃。现在统一进
   `node_outputs`（node_id → outputs），下游按 `depends_on` 取用。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents import AgentContext, BaseAgent, build_agent_registry
from ..config import Settings
from ..contracts import (
    AgentInvocation,
    AgentResult,
    InvocationBudget,
    ResultStatus,
    SubtaskDAG,
    SubtaskNode,
    TaskEvent,
    to_wire,
)
from ..gateway import build_router_from_settings
from ..tools import SqlSandbox

# 业务时区（东八区）。平台面向国内业务，统一用它，避免「上月」算错月份。
TZ_CN = timezone(timedelta(hours=8))

#: Planner 不在 DAG 内（它是图的产出者），但仍要进 node_results/黑板供观测与追溯。
PLANNER_NODE_ID = "planner_1"


# ---------------------------------------------------------------- state reducer


def _merge_dicts(left: dict | None, right: dict | None) -> dict:
    """字典归并：右侧覆盖同名键。用于并行节点写同一状态字段时的合并。"""
    return {**(left or {}), **(right or {})}


def _concat_lists(left: list | None, right: list | None) -> list:
    """列表拼接。用于 node_results 这类「只追加」的字段。"""
    return [*(left or []), *(right or [])]


class PipelineState(TypedDict, total=False):
    """图状态。字段尽量保持可 JSON 序列化，以便 Checkpoint 落库。

    ⚠️ 三个带 reducer 的字段是**并行安全的必要条件**，不要去掉 reducer
    （原因见模块 docstring）。
    """

    # ---- 单次运行的静态上下文：只有入口写入，无并发冲突 ----
    task_event: dict[str, Any]
    now: str
    dag: dict[str, Any]

    # ---- 并发写入的三件套：必须带 reducer ----
    #: node_id -> 该节点的 outputs。下游据此取上游产物。
    node_outputs: Annotated[dict[str, dict[str, Any]], _merge_dicts]
    #: 黑板：key（`<node_id>.<output>`）-> 条目
    blackboard: Annotated[dict[str, Any], _merge_dicts]
    #: 逐节点结果（按完成顺序追加，只增不改）
    node_results: Annotated[list[dict[str, Any]], _concat_lists]


class PipelineNodeFailed(Exception):
    """节点执行失败，终止本次运行。"""

    def __init__(self, node_id: str, code: str, message: str, retryable: bool) -> None:
        self.node_id = node_id
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(f"节点 {node_id} 失败 [{code}]: {message}")


class PipelinePlanError(Exception):
    """Planner 产出的 DAG 无法执行（引用了未注册的角色等）。"""


@dataclass
class PipelineOutcome:
    task_id: str
    status: str  # ok | failed
    blackboard: dict[str, Any] = field(default_factory=dict)
    node_results: list[dict[str, Any]] = field(default_factory=list)
    #: node_id -> outputs。下游取值请走这里或 `outputs_for_role`，不要按固定 node_id 找。
    node_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: str = ""
    dag: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None
    #: 逐次模型调用明细（按发生顺序）。供分级路由的成本/时延归因与 W10 成本计量使用。
    #: 注意：这是**内部结构**，不属于任何已冻结的契约。
    call_records: list[dict[str, Any]] = field(default_factory=list)

    def outputs_for_role(self, role: str) -> list[dict[str, Any]]:
        """按 agent_role 取该角色所有节点的 outputs（按发生顺序，结果稳定）。

        动态 DAG 下节点 ID 是运行时可变的，**按 role 取值才是稳定的**；
        评测/接口/CLI 都走这个入口，而不是按固定 node_id 去找。
        """
        return [
            res.get("outputs") or {}
            for res in self.node_results
            if res.get("agent_role") == role
        ]

    def merged_outputs_for_role(self, role: str) -> dict[str, Any]:
        """按 role 合并该角色所有节点的 outputs（后者覆盖同名键）。

        单节点角色（如 analyst）等价于取该节点 outputs；多节点角色（如两路
        retriever）则是把两路合并——正是下游需要的形态。
        """
        merged: dict[str, Any] = {}
        for outputs in self.outputs_for_role(role):
            merged.update(outputs)
        return merged

    def first_output_for_role(self, role: str) -> dict[str, Any]:
        found = self.outputs_for_role(role)
        return found[0] if found else {}

    @property
    def tokens_by_route(self) -> dict[str, dict[str, int]]:
        """按调用点汇总次数/token/耗时——分级路由的归因入口。"""
        summary: dict[str, dict[str, int]] = {}
        for record in self.call_records:
            key = record.get("route") or "(未标注)"
            bucket = summary.setdefault(
                key,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_ms": 0,
                },
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += record.get("input_tokens", 0)
            bucket["output_tokens"] += record.get("output_tokens", 0)
            bucket["total_tokens"] += record.get("total_tokens", 0)
            bucket["duration_ms"] += record.get("duration_ms", 0)
        return summary

    @property
    def total_input_tokens(self) -> int:
        return sum(r.get("tokens_used", {}).get("input", 0) for r in self.node_results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.get("tokens_used", {}).get("output", 0) for r in self.node_results)


def sink_node_ids(dag: dict[str, Any]) -> list[str]:
    """汇点：没有被任何其它节点依赖的节点。取最终结论就看它们。"""
    if not isinstance(dag, dict):
        # 契约里 dag 是对象；这里显式报错而不是让 "str has no attribute get" 这种
        # 信息量极低的异常冒出去（曾因此排查了一轮间歇性崩溃）
        raise TypeError(f"dag 期望是对象，实际是 {type(dag).__name__}: {dag!r:.120}")
    nodes = dag.get("nodes", [])
    depended_on: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise TypeError(
                f"dag.nodes 的每个元素都应是对象，遇到 {type(node).__name__}: {node!r:.120}"
            )
        depended_on.update(node.get("depends_on", []) or [])
    return [n["node_id"] for n in nodes if n["node_id"] not in depended_on]


class AnalysisPipeline:
    """按 Planner 产出的 DAG 执行分析任务。

    `on_event` 是可选的事件回调（async），用于向外部（如 SSE 接口）推送节点级进度。
    **事件通道故障不得影响主流程**，故回调异常一律吞掉——进度是观测手段，
    不是业务本身。
    """

    def __init__(self, settings: Settings, on_event=None) -> None:
        self.settings = settings
        self._on_event = on_event

    async def _emit(self, event: str, **payload: Any) -> None:
        if self._on_event is None:
            return
        try:
            await self._on_event(event, payload)
        except Exception:
            pass

    # ------------------------------------------------------------------ 图构建
    def _make_invocation(
        self,
        node_id: str,
        role: str,
        inputs: dict[str, Any],
        *,
        task_id: str,
        attempt: int = 1,
    ) -> AgentInvocation:
        return AgentInvocation(
            task_id=task_id,
            node_id=node_id,
            attempt=attempt,
            agent_role=role,
            trace_id=task_id,
            input=inputs,
            budget=InvocationBudget(
                max_tokens=self.settings.default_max_tokens,
                timeout_seconds=self.settings.default_task_timeout_seconds,
            ),
        )

    @staticmethod
    def _build_node_input(node: SubtaskNode, state: PipelineState) -> dict[str, Any]:
        """组装节点输入：问题 + **直接上游**产物（按 key 合并）+ 该节点的声明。

        采用「上游产物按 key 合并」而不是让 Planner 写死数据流接线，是为了让
        Planner 只关心「做什么」，不必关心「数据怎么传」。上游 key 互不重叠时
        合并是安全的；即便重叠，黑板里的 `<node_id>.<output>` 仍保留完整出处。
        """
        merged: dict[str, Any] = {}
        for dep in node.depends_on:
            merged.update(state.get("node_outputs", {}).get(dep, {}))

        return {
            **merged,
            **(node.inputs or {}),
            # 保留键放在最后，保证不被上游产物或节点声明覆盖
            "question": state["task_event"]["payload"]["text"],
            "now": state.get("now", ""),
            "acceptance": list(node.acceptance),
            "node_id": node.node_id,
            "upstream_nodes": list(node.depends_on),
        }

    @staticmethod
    def _absorb(inv: AgentInvocation, result: AgentResult, agent: BaseAgent) -> dict:
        """把节点结果转成**状态增量**；失败则中止。

        只返回本节点的增量（不读旧状态、不做全量覆盖），这样 reducer 才能安全
        合并并行节点的写入。
        """
        if result.status is not ResultStatus.ok:
            err = result.error
            raise PipelineNodeFailed(
                node_id=inv.node_id,
                code=err.code if err else "UNKNOWN",
                message=err.message if err else "未知错误",
                retryable=err.retryable if err else False,
            )

        wire = {
            **to_wire(result),
            # 把角色写进 node_results：下游（评测/接口/CLI）按 role 取值即可，
            # 不必依赖运行时可变的 node_id。node_results 是内部结构，可自由扩展。
            "agent_role": inv.agent_role,
        }
        entry = {
            "version": 1,
            "written_by": agent.labeled_version,  # 用真实 Agent 版本，不再写死
            "confidence": result.confidence,
            "payload": result.outputs,
            "sources": wire.get("sources", []),
        }
        return {
            "node_outputs": {inv.node_id: result.outputs},
            "blackboard": {key: entry for key in result.blackboard_writes},
            "node_results": [wire],
        }

    def _make_node_fn(self, ctx: AgentContext, node: SubtaskNode):
        """为一个 DAG 节点生成 LangGraph 节点函数。"""

        async def run_node(state: PipelineState) -> dict:
            agent = ctx.registry.get(node.agent_role.value)
            if agent is None:  # 建图时已校验，这里只是兜底
                raise PipelinePlanError(
                    f"节点 {node.node_id} 的角色 {node.agent_role.value!r} 未注册"
                )
            task_id = state["task_event"]["task_id"]
            await self._emit(
                "node_started", node_id=node.node_id, agent_role=node.agent_role.value
            )
            inv = self._make_invocation(
                node.node_id,
                node.agent_role.value,
                self._build_node_input(node, state),
                task_id=task_id,
            )
            result = await agent.invoke(inv, ctx)
            delta = self._absorb(inv, result, agent)
            await self._emit(
                "node_finished",
                node_id=node.node_id,
                agent_role=node.agent_role.value,
                status=result.status.value,
                duration_ms=result.duration_ms,
                tokens=result.tokens_used.total,
                knowledge_count=result.outputs.get("knowledge_count", 0),
            )
            return delta

        return run_node

    def _build_from_dag(self, ctx: AgentContext, dag: SubtaskDAG):
        """按 DAG 构图：节点 = DAG 节点，边 = depends_on。

        同一层互不依赖的节点没有互相的入边，LangGraph 会在同一个 superstep 内
        并发执行它们——这正是动态 DAG 引擎要的东西。
        """
        unknown = [
            n.node_id for n in dag.nodes if n.agent_role.value not in ctx.registry
        ]
        if unknown:
            raise PipelinePlanError(
                f"DAG 含未注册角色的节点 {unknown}；已注册角色: {sorted(ctx.registry)}。"
                "请在 agents/build_agent_registry() 中登记后再让 Planner 规划。"
            )

        builder = StateGraph(PipelineState)
        for node in dag.nodes:
            builder.add_node(node.node_id, self._make_node_fn(ctx, node))
        for node in dag.nodes:
            if node.depends_on:
                for dep in node.depends_on:
                    builder.add_edge(dep, node.node_id)
            else:
                builder.add_edge(START, node.node_id)
        for sink in sink_node_ids(dag.model_dump(mode="json")):
            builder.add_edge(sink, END)
        return builder

    # -------------------------------------------------------------------- 执行
    async def run(self, event: TaskEvent, *, resume: bool = False) -> PipelineOutcome:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        started = time.perf_counter()
        task_id = str(event.task_id)
        # 用路由器而非单条主备链：按调用点分派到不同档位（分级路由）
        router = build_router_from_settings(self.settings)
        registry = build_agent_registry()
        ctx = AgentContext(
            settings=self.settings,
            gateway=router,
            sandbox=SqlSandbox(self.settings),
            registry=registry,
        )
        config = {"configurable": {"thread_id": task_id}}

        pool = AsyncConnectionPool(
            conninfo=self.settings.postgres_dsn,
            max_size=4,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open()
        try:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()

            # ---- Planner 先跑（图外的产出者）+ 构图 ----
            if resume:
                # 续跑时图结构必须与首次一致，否则与 checkpoint 的语义未定义。
                # 因此从 checkpoint 里取回原 DAG，而不是重新规划。
                snapshot = await saver.aget(config)
                if snapshot is None:
                    raise PipelineNodeFailed(
                        PLANNER_NODE_ID, "NO_CHECKPOINT", f"线程 {task_id} 无 checkpoint，无法续跑", False
                    )
                dag = SubtaskDAG.model_validate(
                    snapshot.checkpoint["channel_values"]["dag"]
                )
                initial: PipelineState | None = None  # None = 由 checkpoint 恢复状态
            else:
                dag, initial = await self._plan(ctx, event, task_id)

            graph = self._build_from_dag(ctx, dag).compile(checkpointer=saver)
            final = await graph.ainvoke(initial, config=config)
        except PipelineNodeFailed as exc:
            return PipelineOutcome(
                task_id=task_id,
                status="failed",
                error=f"[{exc.code}] {exc.message}",
                duration_ms=int((time.perf_counter() - started) * 1000),
                call_records=_serialize_call_records(router),
            )
        except PipelinePlanError as exc:
            return PipelineOutcome(
                task_id=task_id,
                status="failed",
                error=f"[PLAN_INVALID] {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
                call_records=_serialize_call_records(router),
            )
        finally:
            await pool.close()

        duration_ms = int((time.perf_counter() - started) * 1000)
        node_results = final.get("node_results", [])
        node_outputs = final.get("node_outputs", {})
        outcome = PipelineOutcome(
            task_id=task_id,
            status="ok",
            blackboard=final.get("blackboard", {}),
            node_results=node_results,
            node_outputs=node_outputs,
            summary=_sink_summary(final.get("dag", {}), node_outputs),
            dag=final.get("dag", {}),
            total_tokens=sum(r.get("tokens_used", {}).get("total", 0) for r in node_results),
            duration_ms=duration_ms,
            call_records=_serialize_call_records(router),
        )
        if unmapped := getattr(router, "unmapped_routes", None):
            # 路由表漏配不是致命错误（会退回默认档），但必须能被发现
            outcome.error = f"路由表未配置的调用点: {sorted(unmapped)}"
        return outcome

    async def _plan(
        self, ctx: AgentContext, event: TaskEvent, task_id: str
    ) -> tuple[SubtaskDAG, PipelineState]:
        """图外先跑 Planner，返回校验过的 DAG 与图执行的初始状态。"""
        planner = ctx.registry.get("planner")
        if planner is None:
            raise PipelinePlanError("注册表中缺少 planner，无法规划任务图")

        await self._emit("node_started", node_id=PLANNER_NODE_ID, agent_role="planner")
        inv = self._make_invocation(
            PLANNER_NODE_ID,
            "planner",
            {
                "question": event.payload.text,
                "schema": ctx.sandbox.schema_as_prompt_text(),
                "now": datetime.now(TZ_CN).isoformat(timespec="seconds"),
                # 让 Planner 知道哪些角色真的可执行，避免规划出跑不了的节点
                "available_roles": sorted(
                    role for role in ctx.registry if role != "planner"
                ),
            },
            task_id=task_id,
        )
        result = await planner.invoke(inv, ctx)
        delta = self._absorb(inv, result, planner)  # 失败会抛 PipelineNodeFailed
        await self._emit(
            "node_finished",
            node_id=PLANNER_NODE_ID,
            agent_role="planner",
            status=result.status.value,
            duration_ms=result.duration_ms,
            tokens=result.tokens_used.total,
        )

        dag = SubtaskDAG.model_validate(result.outputs["dag"])
        initial: PipelineState = {
            "task_event": to_wire(event),
            "now": datetime.now(TZ_CN).isoformat(timespec="seconds"),
            "dag": result.outputs["dag"],
            "node_outputs": delta["node_outputs"],
            "blackboard": delta["blackboard"],
            "node_results": delta["node_results"],
        }
        return dag, initial


def _sink_summary(dag: dict[str, Any], node_outputs: dict[str, dict[str, Any]]) -> str:
    """取最终结论：优先汇点节点的 summary，退化到任意节点的 summary。

    动态 DAG 下不该把「谁产出结论」写死在某个 node_id 上。
    """
    for node_id in sink_node_ids(dag):
        outputs = node_outputs.get(node_id)
        if outputs is None:
            continue
        if not isinstance(outputs, dict):
            raise TypeError(
                f"节点 {node_id} 的 outputs 期望是对象，实际是 {type(outputs).__name__}；"
                "AgentResult 契约要求 outputs 为 dict。"
            )
        if outputs.get("summary"):
            return str(outputs["summary"])

    for node_id, outputs in node_outputs.items():
        if isinstance(outputs, dict) and outputs.get("summary"):
            return str(outputs["summary"])
    return ""


def _serialize_call_records(router) -> list[dict[str, Any]]:
    """把网关的逐次调用记录转为可序列化字典（按发生顺序）。"""
    return [asdict(record) for record in getattr(router, "history", [])]


def new_task_event(question: str, *, deadline_minutes: int = 5) -> TaskEvent:
    """构造一个 REST 渠道的 TaskEvent，便于本地跑链路。"""
    from ..contracts import Budget, Payload, Sensitivity, Source, SourceChannel, User, UserRole

    now = datetime.now(TZ_CN)
    return TaskEvent(
        task_id=uuid.uuid4(),
        source=Source(channel=SourceChannel.rest_api),
        submitted_at=now,
        user=User(user_id="local_dev", roles=[UserRole.business_user]),
        payload=Payload(text=question),
        sensitivity=Sensitivity.internal,
        budget=Budget(deadline=now + timedelta(minutes=deadline_minutes)),
    )
