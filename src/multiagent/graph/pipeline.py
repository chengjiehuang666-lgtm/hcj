"""LangGraph 串行链路：Planner → Retriever → Analyst。

W2 的出口标准（见《实施路线图》）：
- 基于 LangGraph 实现三个 Agent 的串行链路
- 任务状态 Checkpoint 落 PostgreSQL，支持断点续跑

设计说明：
- **状态外置**：图的运行态由 LangGraph 的 PostgresSaver 持久化，进程重启后
  凭 thread_id（即 task_id）即可续跑，这是 Product 需求里「会话断点续接」的基础。
- **黑板**：节点之间不互相传私有状态，产物统一写入 state 里的 blackboard，
  与《产品需求》1.3「单一事实源」约定一致。
- **失败即中止**：任一节点返回 status=failed 就抛 PipelineNodeFailed 终止本次
  运行。重试、降级与动态重规划属于 W7 范畴。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents import AgentContext, AnalystAgent, PlannerAgent, RetrieverAgent
from ..config import Settings
from ..contracts import (
    AgentInvocation,
    AgentResult,
    InvocationBudget,
    ResultStatus,
    TaskEvent,
    to_wire,
)
from ..gateway import build_router_from_settings
from ..tools import SqlSandbox

# 业务时区（东八区）。平台面向国内业务，统一用它，避免「上月」算错月份。
TZ_CN = timezone(timedelta(hours=8))


class PipelineState(TypedDict, total=False):
    """图状态。字段尽量保持可 JSON 序列化，以便 Checkpoint 落库。"""

    task_event: dict[str, Any]
    now: str
    schema: str
    data_ranges: str
    metric_definitions: dict[str, Any]
    dag: dict[str, Any]
    blackboard: dict[str, Any]
    node_results: list[dict[str, Any]]
    summary: str


class PipelineNodeFailed(Exception):
    """节点执行失败，终止本次运行。"""

    def __init__(self, node_id: str, code: str, message: str, retryable: bool) -> None:
        self.node_id = node_id
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(f"节点 {node_id} 失败 [{code}]: {message}")


@dataclass
class PipelineOutcome:
    task_id: str
    status: str  # ok | failed
    blackboard: dict[str, Any] = field(default_factory=dict)
    node_results: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    dag: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None
    #: 逐次模型调用明细（按发生顺序）。供分级路由的成本/时延归因与 W10 成本计量使用。
    #: 注意：这是**内部结构**，不属于任何已冻结的契约。
    call_records: list[dict[str, Any]] = field(default_factory=list)

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


class AnalysisPipeline:
    """把用户需求跑成分析报告的串行链路。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
    def _absorb(state: PipelineState, inv: AgentInvocation, result: AgentResult) -> dict:
        """把节点结果合并进状态；失败则中止。"""
        if result.status is not ResultStatus.ok:
            err = result.error
            raise PipelineNodeFailed(
                node_id=inv.node_id,
                code=err.code if err else "UNKNOWN",
                message=err.message if err else "未知错误",
                retryable=err.retryable if err else False,
            )

        wire = to_wire(result)
        blackboard = dict(state.get("blackboard", {}))
        for key in result.blackboard_writes:
            payload = result.outputs
            blackboard[key] = {
                "version": 1,
                "written_by": f"{inv.agent_role}@0.1.0",
                "confidence": result.confidence,
                "payload": payload,
                "sources": wire.get("sources", []),
            }

        merge: dict[str, Any] = {
            "blackboard": blackboard,
            "node_results": [*state.get("node_results", []), wire],
        }
        if inv.agent_role == "planner":
            merge["dag"] = result.outputs.get("dag", {})
        elif inv.agent_role == "retriever":
            merge["schema"] = result.outputs.get("schema", "")
            merge["metric_definitions"] = result.outputs.get("metric_definitions", {})
            merge["data_ranges"] = result.outputs.get("data_ranges", "")
        elif inv.agent_role == "analyst":
            merge["summary"] = result.outputs.get("summary", "")
        return merge

    def _build(self, ctx: AgentContext):
        planner, retriever, analyst = PlannerAgent(), RetrieverAgent(), AnalystAgent()

        async def plan_node(state: PipelineState) -> dict:
            task_id = state["task_event"]["task_id"]
            question = state["task_event"]["payload"]["text"]
            schema = ctx.sandbox.schema_as_prompt_text()
            inv = self._make_invocation(
                "planner_1",
                "planner",
                {"question": question, "schema": schema, "now": state["now"]},
                task_id=task_id,
            )
            result = await planner.invoke(inv, ctx)
            return self._absorb(state, inv, result)

        async def retrieve_node(state: PipelineState) -> dict:
            task_id = state["task_event"]["task_id"]
            question = state["task_event"]["payload"]["text"]
            inv = self._make_invocation(
                "retriever_1",
                "retriever",
                {"question": question, "dag": state.get("dag", {})},
                task_id=task_id,
            )
            result = await retriever.invoke(inv, ctx)
            return self._absorb(state, inv, result)

        async def analyze_node(state: PipelineState) -> dict:
            task_id = state["task_event"]["task_id"]
            question = state["task_event"]["payload"]["text"]
            # 把 Planner 为该节点写的验收标准传给取数环节：
            # 否则「写 SQL 的一步」不知道「下结论的一步」需要哪些维度和基期数据。
            acceptance = [
                criterion
                for node in (state.get("dag") or {}).get("nodes", [])
                if node.get("node_id") == "analyst_1"
                for criterion in node.get("acceptance", [])
            ]
            inv = self._make_invocation(
                "analyst_1",
                "analyst",
                {
                    "question": question,
                    "schema": state.get("schema", ""),
                    "data_ranges": state.get("data_ranges", ""),
                    "metric_definitions": state.get("metric_definitions", {}),
                    "now": state["now"],
                    "acceptance": acceptance,
                },
                task_id=task_id,
            )
            result = await analyst.invoke(inv, ctx)
            return self._absorb(state, inv, result)

        builder = StateGraph(PipelineState)
        builder.add_node("plan", plan_node)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("analyze", analyze_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "analyze")
        builder.add_edge("analyze", END)
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
        ctx = AgentContext(
            settings=self.settings,
            gateway=router,
            sandbox=SqlSandbox(self.settings),
        )
        graph = self._build(ctx)

        initial: PipelineState = {
            "task_event": to_wire(event),
            "now": datetime.now(TZ_CN).isoformat(timespec="seconds"),
            "blackboard": {},
            "node_results": [],
        }

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
            app = graph.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": task_id}}
            # resume=True 时不传初值，由 Checkpoint 恢复上次中断的状态
            payload = None if resume else initial
            final = await app.ainvoke(payload, config=config)
        except PipelineNodeFailed as exc:
            return PipelineOutcome(
                task_id=task_id,
                status="failed",
                error=f"[{exc.code}] {exc.message}",
                duration_ms=int((time.perf_counter() - started) * 1000),
                call_records=_serialize_call_records(router),
            )
        finally:
            await pool.close()

        duration_ms = int((time.perf_counter() - started) * 1000)
        node_results = final.get("node_results", [])
        outcome = PipelineOutcome(
            task_id=task_id,
            status="ok",
            blackboard=final.get("blackboard", {}),
            node_results=node_results,
            summary=final.get("summary", ""),
            dag=final.get("dag", {}),
            total_tokens=sum(r.get("tokens_used", {}).get("total", 0) for r in node_results),
            duration_ms=duration_ms,
            call_records=_serialize_call_records(router),
        )
        if unmapped := getattr(router, "unmapped_routes", None):
            # 路由表漏配不是致命错误（会退回默认档），但必须能被发现
            outcome.error = f"路由表未配置的调用点: {sorted(unmapped)}"
        return outcome


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
