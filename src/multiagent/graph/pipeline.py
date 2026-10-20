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

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from ..agents import AgentContext, BaseAgent, build_agent_registry
from ..config import Settings, get_settings
from ..contracts import (
    AgentInvocation,
    AgentResult,
    BlackboardEntry,
    EntryStatus,
    InvocationBudget,
    ResultStatus,
    SourceRef,
    SubtaskDAG,
    SubtaskNode,
    TaskEvent,
    WrittenBy,
    to_wire,
)
from ..gateway import build_router_from_settings
from ..memory import MemoryStore
from ..observability import log_event, node_context, set_task_id, summarize_calls
from ..security import scan_input
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


def _add_int(left: int | None, right: int | None) -> int:
    """整数累加。用于步骤/Token 计数——并行节点各写各的增量，由 reducer 汇总。"""
    return (left or 0) + (right or 0)


class PipelineState(TypedDict, total=False):
    """图状态。字段尽量保持可 JSON 序列化，以便 Checkpoint 落库。

    ⚠️ 带 reducer 的字段是**并行安全的必要条件**，不要去掉 reducer
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

    # ---- 任务级预算计数：并行节点各写增量，必须带 reducer ----
    #: 已执行步数（节点执行次数，含重试的 attempt；Planner 图外调用计 1 步）
    steps_used: Annotated[int, _add_int]
    #: 已消耗 token（按节点结果累计；失败 attempt 也计入）
    tokens_used: Annotated[int, _add_int]

    # ---- 人工审批（HITL）----
    #: node_id -> 审批决策（approved / rejected / rewritten）
    approvals: Annotated[dict[str, dict[str, Any]], _merge_dicts]
    #: 本图的 DAG 指纹：恢复前比对，拓扑变了就明确拒绝（否则暂停状态会静默失效）
    dag_hash: str


#: 审批门节点名的后缀：`{node_id}__approval`
APPROVAL_SUFFIX = "__approval__"


class ApprovalDecision(BaseModel):
    """人工审批决策（HITL）。rewrite 时 outputs 为替换后的节点产物。"""

    decision: Literal["approved", "rejected", "rewritten"]
    comment: str = ""
    outputs: dict[str, Any] | None = None


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
    #: 本次任务沉淀进长期记忆的条目 id（未沉淀为 None）；沉淀失败时 error 记录原因
    memory_sunk: int | None = None
    memory_error: str | None = None
    #: 人工审批请求（status="awaiting_approval" 时给出：节点、产物、验收标准）
    approval_requests: list[dict[str, Any]] = field(default_factory=list)
    #: 人工审批决策记录：node_id -> 决策（approved / rejected / rewritten）
    approvals: dict[str, Any] = field(default_factory=dict)
    #: 稳定错误码（failed 时给出：BUDGET_EXCEEDED / STEP_LIMIT / DEADLINE_EXCEEDED 等）
    error_code: str | None = None
    #: 该失败是否可重试（契约语义，供上层决定重试/续跑）
    retryable: bool | None = None
    #: 本次执行使用的 DAG 版本（1 = 首版；重规划后递增）
    revision: int = 1
    #: W10 成本计量：按调用点 / Agent 汇总（token、成本、耗时）
    cost_by_route: dict[str, dict[str, Any]] = field(default_factory=dict)
    cost_by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_cost: float = 0.0

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
        visible_blackboard_keys: list[str] | None = None,
        budget: Any = None,
    ) -> AgentInvocation:
        """构造 AgentInvocation；预算**派生自任务预算且只能收紧**（冻结契约语义）。

        独立复核发现：此前直接取 Settings 默认值，提交方在 TaskEvent.budget 里
        收紧的额度被忽略（甚至比提交方要求的更宽）。预算的「熔断执行」属 W7，
        但「逐级传递、只能收紧」是已冻结的契约语义，不在 W7。
        """
        max_tokens = self.settings.default_max_tokens
        timeout_seconds = self.settings.default_task_timeout_seconds
        if budget is not None:
            # budget 可能是 Budget 契约对象（Planner 在图外跑时）或 state 里的
            # 序列化 dict（图内节点——state 必须可 JSON 序列化，存不了对象）。
            if isinstance(budget, dict):
                event_tokens = int(budget.get("max_tokens") or max_tokens)
                deadline = budget.get("deadline")
                if isinstance(deadline, str):
                    try:
                        deadline = datetime.fromisoformat(deadline)
                    except ValueError:
                        deadline = None
            else:
                event_tokens = int(getattr(budget, "max_tokens", None) or max_tokens)
                deadline = getattr(budget, "deadline", None)
            max_tokens = min(max_tokens, event_tokens)
            if isinstance(deadline, datetime):
                try:
                    remaining = int((deadline - datetime.now(TZ_CN)).total_seconds())
                except TypeError:  # 无时区的 deadline：不敢猜，保持默认
                    remaining = 0
                if remaining > 0:
                    timeout_seconds = min(timeout_seconds, remaining)
        return AgentInvocation(
            task_id=task_id,
            node_id=node_id,
            attempt=attempt,
            agent_role=role,
            trace_id=task_id,
            input=inputs,
            budget=InvocationBudget(
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            ),
            visible_blackboard_keys=list(visible_blackboard_keys or []),
        )

    @staticmethod
    def _build_node_input(node: SubtaskNode, state: PipelineState) -> dict[str, Any]:
        """组装节点输入：问题 + **直接上游**产物（按 key 合并）+ 该节点的声明。

        采用「上游产物按 key 合并」而不是让 Planner 写死数据流接线，是为了让
        Planner 只关心「做什么」，不必关心「数据怎么传」。上游 key 互不重叠时
        合并是安全的；即便重叠，黑板里的 `<node_id>.<output>` 仍保留完整出处。

        除产物外，还交付两样**订阅内容**：
        - `blackboard`：直接上游写入的黑板条目（含 version/status/sources）——
          核验型节点（reviewer）要按条目核对证据与版本，光有产物不够；
        - `upstream_sources`：按上游节点分组的来源明细。
        """
        merged: dict[str, Any] = {}
        for dep in node.depends_on:
            merged.update(state.get("node_outputs", {}).get(dep, {}))

        deps = set(node.depends_on)
        visible = {
            key: entry
            for key, entry in (state.get("blackboard") or {}).items()
            # 黑板 key 的命名约定是 <node_id>.<output>（契约正则亦如此）
            if key.split(".", 1)[0] in deps
        }
        upstream_sources = {
            r.get("node_id"): r.get("sources", [])
            for r in (state.get("node_results") or [])
            if r.get("node_id") in deps
        }

        return {
            **merged,
            **(node.inputs or {}),
            # 保留键放在最后，保证不被上游产物或节点声明覆盖
            "question": state["task_event"]["payload"]["text"],
            "now": state.get("now", ""),
            "acceptance": list(node.acceptance),
            "node_id": node.node_id,
            "upstream_nodes": list(node.depends_on),
            "blackboard": visible,
            "upstream_sources": upstream_sources,
        }

    @staticmethod
    def _new_entry(
        key: str,
        inv: AgentInvocation,
        result: AgentResult,
        agent: BaseAgent,
        written_at: datetime,
    ) -> BlackboardEntry:
        """构造契约化黑板条目（`BlackboardEntry`），首次写入版本号为 1。"""
        return BlackboardEntry(
            task_id=inv.task_id,
            key=key,
            version=1,
            payload=result.outputs,
            written_by=WrittenBy(
                agent_role=inv.agent_role,
                agent_version=agent.version,
                node_id=inv.node_id,
                attempt=inv.attempt,
            ),
            written_at=written_at,
            # 新写入一律 draft：经 Reviewer 核验后才流转为 validated（见契约 status 语义）
            status=EntryStatus.draft,
            sources=[
                SourceRef.model_validate(s) for s in (to_wire(result).get("sources") or [])
            ],
            confidence=result.confidence,
        )

    @staticmethod
    def _dump_entry(entry: BlackboardEntry) -> dict[str, Any]:
        """条目出网序列化：走 `to_wire`（exclude_none）。

        不能用裸 `model_dump(mode="json")`：冻结 schema 里 `confidence` 只允许
        number（无 null 分支），None 会序列化成 `"confidence": null` 被冻结契约拒绝
        ——而 AgentResult.confidence 本身是可选的合法形态（独立复核实测复现）。
        """
        return to_wire(entry)

    @staticmethod
    def _apply_blackboard_updates(
        current: dict[str, Any],
        updates: list[Any],
        inv: AgentInvocation,
        agent: BaseAgent,
    ) -> dict[str, dict[str, Any]]:
        """对既有条目做**乐观锁更新**——契约语义：提交读到的 version，比对不符即拒绝。

        更新意图由 Agent 在 outputs 的保留键 `blackboard_updates` 声明：
            [{"key": "analyst_1.report", "expected_version": 1, "status": "validated"}]
        拒绝而不静默覆盖：状态流转（draft → validated/rejected）是
        「产物经 Reviewer 抽检」的落点，让它悄悄错版等于废掉审核语义。
        """
        applied: dict[str, dict[str, Any]] = {}
        for item in updates:
            if not isinstance(item, dict) or "key" not in item or "expected_version" not in item:
                raise PipelineNodeFailed(
                    node_id=inv.node_id,
                    code="BAD_BLACKBOARD_UPDATE",
                    message=f"blackboard_updates 元素形状非法: {item!r:.200}",
                    retryable=False,
                )
            key = str(item["key"])
            raw = current.get(key)
            if raw is None:
                raise PipelineNodeFailed(
                    node_id=inv.node_id,
                    code="BLACKBOARD_CONFLICT",
                    message=f"更新了不存在的黑板条目 {key}",
                    retryable=False,
                )
            entry = BlackboardEntry.model_validate(raw)
            expected = int(item["expected_version"])
            if entry.version != expected:
                raise PipelineNodeFailed(
                    node_id=inv.node_id,
                    code="BLACKBOARD_CONFLICT",
                    message=(
                        f"条目 {key} 版本冲突：写入方读到 v{expected}，当前 v{entry.version}；"
                        "写入被拒绝（乐观锁）"
                    ),
                    retryable=True,
                )
            try:
                new_status = EntryStatus(str(item["status"]))
            except ValueError as exc:
                raise PipelineNodeFailed(
                    node_id=inv.node_id,
                    code="BAD_BLACKBOARD_UPDATE",
                    message=f"非法条目状态 {item['status']!r}：{exc}",
                    retryable=False,
                ) from exc
            updated = entry.next_version(
                entry.payload,
                status=new_status,
                written_by=WrittenBy(
                    agent_role=inv.agent_role,
                    agent_version=agent.version,
                    node_id=inv.node_id,
                    attempt=inv.attempt,
                ),
                written_at=datetime.now(TZ_CN),
            )
            applied[key] = AnalysisPipeline._dump_entry(updated)
        return applied

    @staticmethod
    def _absorb(
        inv: AgentInvocation,
        result: AgentResult,
        agent: BaseAgent,
        *,
        blackboard: dict[str, Any] | None = None,
        written_at: datetime | None = None,
    ) -> dict:
        """把节点结果转成**状态增量**；失败则中止。

        只返回本节点的增量（不读旧状态、不做全量覆盖），这样 reducer 才能安全
        合并并行节点的写入。

        黑板条目按**冻结契约** `BlackboardEntry` 构造，不再手写近似 dict——
        条目是「结论可回溯」的落点，形状漂移会让契约校验形同虚设。
        """
        # needs_replan / needs_human 是契约里的**独立语义**，不是 failed：
        # 前者交给 run() 的有界重规划，后者是「转人工」的落点（终止于稳定错误码）。
        if result.status is ResultStatus.needs_replan:
            raise PipelineNodeFailed(
                inv.node_id, "NEEDS_REPLAN", "Agent 判定当前拆解不可行，需重规划", False
            )
        if result.status is ResultStatus.needs_human:
            raise PipelineNodeFailed(
                inv.node_id, "NEEDS_HUMAN", "Agent 请求转人工", False
            )
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
        current = blackboard or {}
        written_at = written_at or datetime.now(TZ_CN)
        entries: dict[str, dict[str, Any]] = {}
        for key in result.blackboard_writes:
            if key in current:
                raise PipelineNodeFailed(
                    node_id=inv.node_id,
                    code="BLACKBOARD_CONFLICT",
                    message=(
                        f"键 {key} 已存在（v{current[key].get('version')}）；"
                        "覆盖既有条目必须走 blackboard_updates（乐观锁）"
                    ),
                    retryable=False,
                )
            entry = AnalysisPipeline._new_entry(key, inv, result, agent, written_at)
            entries[key] = AnalysisPipeline._dump_entry(entry)

        updates = (result.outputs or {}).get("blackboard_updates") or []
        if updates:
            entries.update(
                AnalysisPipeline._apply_blackboard_updates(current, updates, inv, agent)
            )

        return {
            "node_outputs": {inv.node_id: result.outputs},
            "blackboard": entries,
            "node_results": [wire],
        }

    @staticmethod
    def _budget_limits(state: PipelineState) -> dict[str, Any]:
        """取任务预算（state 里是序列化 dict）；缺项回退契约默认值。"""
        raw = (state.get("task_event") or {}).get("budget") or {}
        return {
            "max_steps": int(raw.get("max_steps") or 30),
            "max_tokens": int(raw.get("max_tokens") or 200_000),
            "max_retries": int(raw.get("max_retries") or 0),
            "deadline": raw.get("deadline"),
        }

    @staticmethod
    def _budget_breach(node_id: str, state: PipelineState) -> tuple[str, str] | None:
        """执行**前**检查四大约束；超限返回 (code, message)，否则 None。

        为什么不挪到节点完成后检查：LangGraph 的兄弟节点在同一 superstep 里
        并发起跑，事后检查只能拦住「下一个」节点；事前检查能让超限后**不再启动
        新节点**。代价是单节点的开销可能使总量略微越界，这是可接受的边界。

        「转人工」的落点：任务以稳定错误码终止 + `task_breaker_tripped` 事件
        （单租户无审批队列；人工通过 GET /tasks 查看原因）。
        """
        limits = AnalysisPipeline._budget_limits(state)
        steps = int(state.get("steps_used") or 0)
        tokens = int(state.get("tokens_used") or 0)
        if steps >= limits["max_steps"]:
            return (
                "STEP_LIMIT",
                f"节点 {node_id} 启动前已执行 {steps} 步，达到 max_steps={limits['max_steps']}",
            )
        if tokens >= limits["max_tokens"]:
            return (
                "BUDGET_EXCEEDED",
                f"节点 {node_id} 启动前已消耗 {tokens} token，达到 max_tokens={limits['max_tokens']}",
            )
        deadline_raw = limits["deadline"]
        if deadline_raw:
            try:
                deadline = datetime.fromisoformat(str(deadline_raw))
            except ValueError:
                deadline = None
            if deadline is not None:
                try:
                    remaining = (deadline - datetime.now(TZ_CN)).total_seconds()
                except TypeError:  # 无时区的 deadline：不猜
                    remaining = None
                if remaining is not None and remaining <= 0:
                    return (
                        "DEADLINE_EXCEEDED",
                        f"节点 {node_id} 启动前任务已过截止时间 {deadline_raw}",
                    )
        return None

    def _make_node_fn(self, ctx: AgentContext, node: SubtaskNode):
        """为一个 DAG 节点生成 LangGraph 节点函数。"""

        async def run_node(state: PipelineState) -> dict:
            agent = ctx.registry.get(node.agent_role.value)
            if agent is None:  # 建图时已校验，这里只是兜底
                raise PipelinePlanError(
                    f"节点 {node.node_id} 的角色 {node.agent_role.value!r} 未注册"
                )
            task_id = state["task_event"]["task_id"]
            # 人工驳回：上游被驳回后，下游节点在执行前短路（Agent 不会运行）。
            # 用异常而非 Command(goto)：后者不会压制静态边，下游照样会跑（实测）。
            rejected_upstream = [
                nid
                for nid, d in (state.get("approvals") or {}).items()
                if d.get("decision") == "rejected"
            ]
            if rejected_upstream:
                raise PipelineNodeFailed(
                    node.node_id,
                    "APPROVAL_REJECTED",
                    f"上游节点 {rejected_upstream} 被人工驳回，任务终止",
                    False,
                )
            # 四大约束：超限的节点**不启动**（熔断 + 事件）
            if breach := self._budget_breach(node.node_id, state):
                code, message = breach
                limits = self._budget_limits(state)
                await self._emit(
                    "task_breaker_tripped",
                    node_id=node.node_id,
                    code=code,
                    message=message,
                    steps_used=int(state.get("steps_used") or 0),
                    tokens_used=int(state.get("tokens_used") or 0),
                    max_steps=limits["max_steps"],
                    max_tokens=limits["max_tokens"],
                )
                raise PipelineNodeFailed(node.node_id, code, message, False)
            await self._emit(
                "node_started", node_id=node.node_id, agent_role=node.agent_role.value
            )
            inputs = self._build_node_input(node, state)
            limits = self._budget_limits(state)
            # 尝试次数 = 1 + min(节点 max_retries, 任务 max_retries)。
            # 两层重试各归各的：Agent 内部的 call_json 纠错不算在这里，这里只重试
            # **节点级**的可重试失败（契约 error.retryable）。
            max_attempts = 1 + max(0, min(int(node.max_retries), limits["max_retries"]))
            timeout = min(
                self.settings.default_task_timeout_seconds, int(node.timeout_seconds)
            )
            last_error: PipelineNodeFailed | None = None
            delta: dict[str, Any] | None = None
            result = None
            attempt = 1
            while attempt <= max_attempts:
                inv = self._make_invocation(
                    node.node_id,
                    node.agent_role.value,
                    inputs,
                    task_id=task_id,
                    attempt=attempt,
                    # 契约字段落到实处：本节点能读到的黑板条目 = 直接上游写入的那些
                    visible_blackboard_keys=sorted(inputs.get("blackboard") or {}),
                    budget=(state.get("task_event") or {}).get("budget"),
                )
                try:
                    with node_context(node.node_id, node.agent_role.value):
                        result = await asyncio.wait_for(
                            agent.invoke(inv, ctx), timeout=float(timeout)
                        )
                except TimeoutError:
                    last_error = PipelineNodeFailed(
                        node.node_id,
                        "TIMEOUT",
                        f"第 {attempt} 次执行超时（上限 {timeout}s）",
                        True,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — 未预料异常按可重试处理
                    last_error = PipelineNodeFailed(
                        node.node_id,
                        "INTERNAL_ERROR",
                        f"{type(exc).__name__}: {exc}"[:300],
                        True,
                    )
                if result is not None:
                    try:
                        delta = self._absorb(
                            inv, result, agent, blackboard=state.get("blackboard") or {}
                        )
                    except PipelineNodeFailed as exc:
                        last_error = exc
                        if not exc.retryable:
                            raise
                    else:
                        break  # 成功
                await self._emit(
                    "node_retry",
                    node_id=node.node_id,
                    agent_role=node.agent_role.value,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    code=last_error.code if last_error else "UNKNOWN",
                    message=(last_error.message if last_error else "")[:200],
                )
                result = None
                attempt += 1
            if delta is None:
                assert last_error is not None
                raise last_error
            # 预算计数：步数与 token 都计入（并行节点各写增量，由 reducer 汇总）。
            # 注意：失败的 attempt 其 token 体现在 call_records（成本归因），
            # 不计入本 delta——节点抛错时 LangGraph 不落它的写。
            delta["steps_used"] = 1
            delta["tokens_used"] = result.tokens_used.total
            for key, entry in delta.get("blackboard", {}).items():
                # 黑板变更通知走进程内事件通道（= SSE）；跨进程订阅属多进程阶段
                await self._emit(
                    "blackboard_written",
                    key=key,
                    version=entry.get("version"),
                    status=entry.get("status"),
                    written_by=entry.get("written_by"),
                    node_id=node.node_id,
                    agent_role=node.agent_role.value,
                )
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

    def _make_approval_gate(self, node: SubtaskNode):
        """为一个要求人工审批的节点生成审批门（HITL 后端阻塞点）。

        **门是独立节点、位于被审节点之后**：LangGraph 恢复时会把中断所在节点从头
        重放，若在 Agent 节点内 pause，模型调用与黑板写入都会重复执行。门内只有
        interrupt 与纯状态整理，重放无副作用；审批发生在被审节点完成之后的新
        superstep，下游因此不会提前起跑。
        """

        async def gate(state: PipelineState) -> dict:
            outputs = (state.get("node_outputs") or {}).get(node.node_id) or {}
            keys = sorted(
                key
                for key in (state.get("blackboard") or {})
                if key.split(".", 1)[0] == node.node_id
            )
            decision = interrupt(
                {
                    "kind": "node_approval",
                    "task_id": state["task_event"]["task_id"],
                    "node_id": node.node_id,
                    "agent_role": node.agent_role.value,
                    "acceptance": list(node.acceptance),
                    "outputs": outputs,
                    "blackboard_keys": keys,
                },
                response_schema=ApprovalDecision,
            )
            result: dict[str, Any] = {
                "approvals": {node.node_id: decision.model_dump()},
            }
            if decision.decision == "rewritten":
                replacement = dict(decision.outputs or {})
                result["node_outputs"] = {node.node_id: replacement}
                # 人工改写以版本 +1 写入黑板：状态直接 validated（人已确认），
                # 写入者记 system（契约枚举无 human；审批人记在 source ref 里）
                updated: dict[str, Any] = {}
                for key in keys:
                    entry = BlackboardEntry.model_validate(state["blackboard"][key])
                    updated[key] = to_wire(
                        entry.next_version(
                            replacement,
                            status=EntryStatus.validated,
                            written_by=WrittenBy(
                                agent_role="system",
                                agent_version="human-approval",
                                node_id=node.node_id,
                                attempt=1,
                            ),
                            written_at=datetime.now(TZ_CN),
                        )
                    )
                result["blackboard"] = updated
            return result

        return gate

    @staticmethod
    def _dag_hash(dag_json: dict[str, Any]) -> str:
        """DAG 指纹：恢复前比对。拓扑变了而恢复，暂停状态会**静默失效**。"""
        import hashlib
        import json as _json

        payload = _json.dumps(dag_json, ensure_ascii=False, sort_keys=True)
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _build_from_dag(self, ctx: AgentContext, dag: SubtaskDAG):
        """按 DAG 构图：节点 = DAG 节点，边 = depends_on；审批节点后插审批门。

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

        # 审批门：requires_human_approval=true 的节点之后
        gates = {
            n.node_id: f"{n.node_id}{APPROVAL_SUFFIX}"
            for n in dag.nodes
            if n.requires_human_approval
        }
        for node in dag.nodes:
            if node.node_id in gates:
                builder.add_node(gates[node.node_id], self._make_approval_gate(node))
                builder.add_edge(node.node_id, gates[node.node_id])

        dependents: dict[str, list[str]] = {}
        for node in dag.nodes:
            for dep in node.depends_on:
                dependents.setdefault(dep, []).append(node.node_id)

        for node in dag.nodes:
            if node.depends_on:
                # 必须用**列表形式**连边（join 语义）。逐个 add_edge(dep, node) 是
                # 「任一上游完成即触发」：依赖跨层时下游会提前执行一次、依赖齐了再跑一次。
                # 这是 W6 被黑板冲突检测抓到的 W5 引擎缺陷（菱形用例的两路恰好同一步完成，
                # 掩盖了它；只有依赖横跨两层时才现形）。
                # 依赖了被审节点时，边改接到审批门（等人工批准后才放行）。
                builder.add_edge(
                    [gates.get(dep, dep) for dep in node.depends_on], node.node_id
                )
            else:
                builder.add_edge(START, node.node_id)

        # 驳回的两种收尾：被审汇点由 run() 读 approvals 判 rejected；有下游时，
        # 下游节点在执行前看到驳回记录即短路（见 _make_node_fn 顶部），
        # 不依赖 Command(goto) ——实测它不会压制静态边，下游照样会跑。
        gated_sinks = set(gates)
        for sink in sink_node_ids(dag.model_dump(mode="json")):
            builder.add_edge(gates.get(sink, sink) if sink in gated_sinks else sink, END)
        return builder

    # -------------------------------------------------------------------- 执行
    async def run(
        self,
        event: TaskEvent,
        *,
        resume: bool = False,
        resume_value: Any = None,
    ) -> PipelineOutcome:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        started = time.perf_counter()
        task_id = str(event.task_id)
        # W10：task_id 贯穿整个调用链（网关/Agent/工具日志与 CallRecord 自动携带）
        set_task_id(task_id)
        # 输入侧安全兜底（W9）：API 提交前已拦一次；CLI/评测/直调也必须拦，
        # 否则安全基线只在 REST 入口生效。命中时不建图、不连库、不调模型。
        if self.settings.security_enabled:
            verdict = scan_input(
                event.payload.text,
                blocked_terms=self.settings.blocked_terms,
                enabled=True,
            )
            if not verdict.allowed:
                await self._emit(
                    "input_blocked",
                    task_id=task_id,
                    rule=verdict.rule,
                    reason=verdict.blocked_reason,
                )
                return PipelineOutcome(
                    task_id=task_id,
                    status="failed",
                    error=f"[INPUT_BLOCKED] {verdict.blocked_reason}",
                    error_code="INPUT_BLOCKED",
                    retryable=False,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            if verdict.redactions:
                event = event.model_copy(
                    update={
                        "payload": event.payload.model_copy(
                            update={"text": verdict.text}
                        )
                    }
                )
        # 用路由器而非单条主备链：按调用点分派到不同档位（分级路由）
        router = build_router_from_settings(self.settings)
        registry = build_agent_registry()
        ctx = AgentContext(
            settings=self.settings,
            gateway=router,
            sandbox=SqlSandbox(self.settings),
            registry=registry,
        )
        #: 重规划轮次（1 = 首版）。每轮用独立 thread：新 DAG 从头执行，
        #: 不复用旧图状态（复用需要改写 checkpoint，明确不做）。
        revision = 1
        max_replans = max(0, int(self.settings.max_replans))
        carried: dict[str, Any] = {}
        previous_dag: SubtaskDAG | None = None
        last_failure: PipelineNodeFailed | None = None
        config = {"configurable": {"thread_id": task_id}}

        pool = AsyncConnectionPool(
            conninfo=self.settings.postgres_dsn,
            # min_size 必须显式给 1：psycopg_pool 默认 min_size=4，N 个并发任务
            # 会占 4N 条连接（PG max_connections=100 → 约 24 并发即触顶）。
            # 压测（W8）前修正；进程级共享 checkpointer 池属后续优化。
            min_size=1,
            max_size=4,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open()
        try:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()

            while True:
                thread_id = task_id if revision == 1 else f"{task_id}:r{revision}"
                config = {"configurable": {"thread_id": thread_id}}

                # ---- Planner 先跑（图外的产出者）+ 构图 ----
                if resume and revision == 1:
                    # 续跑时图结构必须与首次一致，否则与 checkpoint 的语义未定义。
                    # 因此从 checkpoint 里取回原 DAG，而不是重新规划。
                    snapshot = await saver.aget(config)
                    if snapshot is None:
                        raise PipelineNodeFailed(
                            PLANNER_NODE_ID, "NO_CHECKPOINT", f"线程 {task_id} 无 checkpoint，无法续跑", False
                        )
                    saved = (
                        snapshot.get("channel_values") or {}
                        if isinstance(snapshot, dict)
                        else (getattr(snapshot, "checkpoint", None) or {}).get("channel_values") or {}
                    )
                    dag = SubtaskDAG.model_validate(saved["dag"])
                    # 拓扑守卫：节点/边变了而恢复，LangGraph 会**静默丢弃**暂停状态
                    stored_hash = saved.get("dag_hash")
                    if stored_hash and stored_hash != self._dag_hash(saved["dag"]):
                        raise PipelineNodeFailed(
                            PLANNER_NODE_ID,
                            "DAG_MISMATCH",
                            "恢复时的 DAG 指纹与 checkpoint 不一致（拓扑已被修改）",
                            False,
                        )
                    # 动态中断（HITL）只能用 Command(resume=...) 恢复；
                    # 传 None 是给静态断点用的，会重放节点并再次暂停。
                    initial = (
                        Command(resume=resume_value) if resume_value is not None else None  # type: ignore[assignment]
                    )
                else:
                    dag, initial = await self._plan(
                        ctx,
                        event,
                        task_id,
                        previous=previous_dag,
                        failure=last_failure,
                        revision=revision,
                    )
                    # 预算是**任务级**的：重规划承接前一轮已消耗的步数/token
                    initial["steps_used"] = int(carried.get("steps_used") or 0) + int(
                        initial.get("steps_used") or 0
                    )
                    initial["tokens_used"] = int(carried.get("tokens_used") or 0) + int(
                        initial.get("tokens_used") or 0
                    )

                try:
                    graph = self._build_from_dag(ctx, dag).compile(checkpointer=saver)
                    final = await graph.ainvoke(initial, config=config)
                    break
                except PipelineNodeFailed as exc:
                    state = await self._read_checkpoint_state(saver, config)
                    # 触发重规划的两条路径：Agent 显式 needs_replan；或可重试失败
                    # 已耗尽节点重试。其余错误（含 NEEDS_HUMAN）直接终止。
                    if exc.code == "NEEDS_HUMAN":
                        raise
                    if revision <= max_replans and (
                        exc.code == "NEEDS_REPLAN" or exc.retryable
                    ):
                        await self._emit(
                            "replan_triggered",
                            task_id=task_id,
                            revision=revision + 1,
                            failure_code=exc.code,
                            node_id=exc.node_id,
                            message=exc.message[:200],
                        )
                        previous_dag = dag
                        last_failure = exc
                        carried = {
                            "steps_used": int(state.get("steps_used") or 0),
                            "tokens_used": int(state.get("tokens_used") or 0),
                        }
                        revision += 1
                        continue
                    raise
        except PipelineNodeFailed as exc:
            # 失败路径保真：从 checkpoint 取回已完成节点的结果与累计 token。
            # W6 的观测缺口：失败时 outcome 只带错误串、token 显示 0、已完成节点全丢
            # ——而失败/重试恰恰最耗 token，熔断计数与成本归因都依赖它。
            state = await self._read_checkpoint_state(saver, config)
            return self._failed_outcome(
                task_id, exc, state, router, started, revision=revision
            )
        except PipelinePlanError as exc:
            state = await self._read_checkpoint_state(saver, config)
            return self._failed_outcome(
                task_id,
                PipelineNodeFailed(PLANNER_NODE_ID, "PLAN_INVALID", str(exc), False),
                state,
                router,
                started,
                revision=revision,
            )
        finally:
            await pool.close()

        # HITL：运行被 interrupt 暂停时，ainvoke 正常返回并带 __interrupt__
        if isinstance(final, dict) and final.get("__interrupt__"):
            requests = [
                {"id": item.id, "value": item.value}
                for item in final["__interrupt__"]
            ]
            state = {k: v for k, v in final.items() if k != "__interrupt__"}
            return self._awaiting_outcome(
                task_id, state, requests, router, started, revision=revision
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        node_results = final.get("node_results", [])
        node_outputs = final.get("node_outputs", {})
        approvals = final.get("approvals") or {}
        rejected = sorted(
            nid for nid, d in approvals.items() if d.get("decision") == "rejected"
        )
        outcome = PipelineOutcome(
            task_id=task_id,
            status="rejected" if rejected else "ok",
            blackboard=final.get("blackboard", {}),
            node_results=node_results,
            node_outputs=node_outputs,
            summary=_sink_summary(final.get("dag", {}), node_outputs),
            dag=final.get("dag", {}),
            total_tokens=sum(r.get("tokens_used", {}).get("total", 0) for r in node_results),
            duration_ms=duration_ms,
            call_records=_serialize_call_records(router),
            approvals=approvals,
            error=(
                f"人工驳回：{', '.join(rejected)}" if rejected else None
            ),
            revision=revision,
        )
        if unmapped := getattr(router, "unmapped_routes", None):
            # 路由表漏配不是致命错误（会退回默认档），但必须能被发现
            outcome.error = f"路由表未配置的调用点: {sorted(unmapped)}"
        outcome = self._apply_metering(outcome)
        # 结论沉淀（best-effort）：门槛见 memory.distill_memory（任务成功 + 评审未否决）。
        # 失败只记录不抛出——记忆是增强，不是任务成功的必要条件。
        if self.settings.memory_sink_enabled and outcome.status == "ok":
            await self._sink_memory(outcome)
        return outcome

    def _apply_metering(self, outcome: PipelineOutcome) -> PipelineOutcome:
        """W10：按调用点 / Agent 汇总 token 与成本，并写一条任务级结构化日志。"""
        summary = summarize_calls(
            outcome.call_records, self.settings.llm_cost_per_1k
        )
        outcome.cost_by_route = summary["by_route"]
        outcome.cost_by_agent = summary["by_agent"]
        outcome.total_cost = summary["total_cost"]
        log_event(
            "task_finished",
            status=outcome.status,
            tokens=outcome.total_tokens,
            cost=outcome.total_cost,
            cost_unit=self.settings.cost_unit,
            duration_ms=outcome.duration_ms,
            revision=outcome.revision,
            error_code=outcome.error_code,
        )
        return outcome

    @staticmethod
    async def _read_checkpoint_state(saver, config) -> dict[str, Any]:
        """失败路径：从 checkpoint 取回已完成节点的状态（best-effort）。

        取不到（例如 Planner 阶段就失败、尚无 checkpoint）时返回空 dict，
        失败结果退化为「只带错误信息」——不因为读 checkpoint 失败而吞掉原始错误。
        """
        try:
            snapshot = await saver.aget(config)
        except Exception:  # noqa: BLE001 — 观测增强不得掩盖原始失败
            return {}
        if snapshot is None:
            return {}
        # langgraph-checkpoint 4.3 的 `aget` 返回 checkpoint **dict 本身**；
        # CheckpointTuple（带 .checkpoint）是 `aget_tuple` 的形态。两种都兼容，
        # 免得再踩「按文档假设形状」的坑（W5 的 resume 分支正是因此从未走通）。
        if isinstance(snapshot, dict):
            return snapshot.get("channel_values") or {}
        return (getattr(snapshot, "checkpoint", None) or {}).get("channel_values") or {}

    def _failed_outcome(
        self,
        task_id: str,
        exc: PipelineNodeFailed,
        state: dict[str, Any],
        router,
        started: float,
        *,
        revision: int = 1,
    ) -> PipelineOutcome:
        """构造失败结果，并尽量带上已完成节点的部分结果与 token。"""
        node_results = list(state.get("node_results") or [])
        node_outputs = dict(state.get("node_outputs") or {})
        dag = state.get("dag") or {}
        try:
            summary = _sink_summary(dag, node_outputs) if dag else ""
        except Exception:  # noqa: BLE001 — 部分状态的形状可能不完整
            summary = ""
        status = "rejected" if exc.code == "APPROVAL_REJECTED" else "failed"
        outcome = PipelineOutcome(
            task_id=task_id,
            status=status,
            blackboard=dict(state.get("blackboard") or {}),
            node_results=node_results,
            node_outputs=node_outputs,
            summary=summary,
            dag=dag,
            total_tokens=sum(
                r.get("tokens_used", {}).get("total", 0) for r in node_results
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
            # 带上节点名：只报错误码会让「哪个节点挂的」无从判断
            error=f"[{exc.code}] 节点 {exc.node_id}: {exc.message}",
            call_records=_serialize_call_records(router),
            error_code=exc.code,
            retryable=exc.retryable,
            revision=revision,
        )
        return self._apply_metering(outcome)

    def _awaiting_outcome(
        self,
        task_id: str,
        state: dict[str, Any],
        requests: list[dict[str, Any]],
        router,
        started: float,
        *,
        revision: int = 1,
    ) -> PipelineOutcome:
        """构造「等待人工审批」的暂停结果（非终态）。"""
        node_results = list(state.get("node_results") or [])
        node_outputs = dict(state.get("node_outputs") or {})
        dag = state.get("dag") or {}
        try:
            summary = _sink_summary(dag, node_outputs) if dag else ""
        except Exception:  # noqa: BLE001
            summary = ""
        outcome = PipelineOutcome(
            task_id=task_id,
            status="awaiting_approval",
            blackboard=dict(state.get("blackboard") or {}),
            node_results=node_results,
            node_outputs=node_outputs,
            summary=summary,
            dag=dag,
            total_tokens=sum(
                r.get("tokens_used", {}).get("total", 0) for r in node_results
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
            call_records=_serialize_call_records(router),
            approval_requests=requests,
            revision=revision,
        )
        return self._apply_metering(outcome)

    async def load_saved_task(self, task_id: str) -> TaskEvent | None:
        """从 checkpoint 读回任务的原始 TaskEvent（API 重启后审批/恢复用）。

        thread_id 就是 API 的 task_id（提交时只创建一次 TaskEvent），因此进程重启后
        仍能把「哪个任务、什么预算、什么截止时间」从 PostgreSQL 找回来。
        """
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            conninfo=self.settings.postgres_dsn,
            # max_size=2 时必须显式 min_size≤2：默认 min_size=4 会让构造直接抛
            # ValueError（W7 缺陷：进程重启后的审批恢复因此 404）
            min_size=1,
            max_size=2,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open()
        try:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            state = await self._read_checkpoint_state(
                saver, {"configurable": {"thread_id": task_id}}
            )
        finally:
            await pool.close()
        raw = state.get("task_event")
        if not raw:
            return None
        return TaskEvent.model_validate(raw)

    async def _sink_memory(self, outcome: PipelineOutcome) -> None:
        try:
            memory_id = await asyncio.to_thread(
                lambda: MemoryStore(self.settings).sink_outcome(outcome)
            )
        except Exception as exc:
            outcome.memory_error = f"{type(exc).__name__}: {exc}"[:300]
            await self._emit(
                "memory_sink_failed", task_id=outcome.task_id, error=outcome.memory_error
            )
            return
        if memory_id is not None:
            outcome.memory_sunk = memory_id
            await self._emit(
                "memory_sunk", task_id=outcome.task_id, memory_id=memory_id
            )

    async def _plan(
        self,
        ctx: AgentContext,
        event: TaskEvent,
        task_id: str,
        *,
        previous: SubtaskDAG | None = None,
        failure: PipelineNodeFailed | None = None,
        revision: int = 1,
    ) -> tuple[SubtaskDAG, PipelineState]:
        """图外先跑 Planner，返回校验过的 DAG 与图执行的初始状态。

        `previous/failure/revision` 用于**有界重规划**：把上一版 DAG 与失败原因
        交给 Planner，产出一版新的拆解；`revision` 由代码写死（不要交给模型自由填）。
        """
        planner = ctx.registry.get("planner")
        if planner is None:
            raise PipelinePlanError("注册表中缺少 planner，无法规划任务图")

        await self._emit("node_started", node_id=PLANNER_NODE_ID, agent_role="planner")
        # schema 自省是**同步 psycopg**调用：直接跑会冻结整个事件循环（W5 只修了
        # Retriever）。并发下一条慢查询会让所有任务的 SSE/轮询一起停顿。
        schema_text = await asyncio.to_thread(ctx.sandbox.schema_as_prompt_text)
        inputs: dict[str, Any] = {
            "question": event.payload.text,
            "schema": schema_text,
            "now": datetime.now(TZ_CN).isoformat(timespec="seconds"),
            # 让 Planner 知道哪些角色真的可执行，避免规划出跑不了的节点
            "available_roles": sorted(
                role for role in ctx.registry if role != "planner"
            ),
        }
        if previous is not None and failure is not None:
            inputs["previous_dag"] = previous.model_dump(mode="json")
            inputs["failure_reason"] = (
                f"[{failure.code}] 节点 {failure.node_id}: {failure.message}"
            )
            inputs["revision"] = revision
        inv = self._make_invocation(
            PLANNER_NODE_ID,
            "planner",
            inputs,
            task_id=task_id,
            budget=event.budget,
        )
        result = await planner.invoke(inv, ctx)
        # 失败会抛 PipelineNodeFailed；Planner 在黑板书之前，current 传空
        delta = self._absorb(inv, result, planner, blackboard={})
        await self._emit(
            "node_finished",
            node_id=PLANNER_NODE_ID,
            agent_role="planner",
            status=result.status.value,
            duration_ms=result.duration_ms,
            tokens=result.tokens_used.total,
        )

        dag = SubtaskDAG.model_validate(result.outputs["dag"])
        if revision > 1:
            # revision 由编排层决定，不交给模型自由填（契约：重规划次数，递增可追溯）
            dag = dag.model_copy(update={"revision": revision})
            result.outputs["dag"] = dag.model_dump(mode="json", exclude_none=True)
        initial: PipelineState = {
            "task_event": to_wire(event),
            "now": datetime.now(TZ_CN).isoformat(timespec="seconds"),
            "dag": result.outputs["dag"],
            "node_outputs": delta["node_outputs"],
            "blackboard": delta["blackboard"],
            "node_results": delta["node_results"],
            # Planner 在图外跑，也消耗一步与它的 token
            "steps_used": 1,
            "tokens_used": result.tokens_used.total,
            "dag_hash": self._dag_hash(result.outputs["dag"]),
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


def new_task_event(
    question: str,
    *,
    deadline_minutes: int = 5,
    settings: Settings | None = None,
    task_id: str | None = None,
) -> TaskEvent:
    """构造一个 REST 渠道的 TaskEvent，便于本地跑链路。

    预算从 Settings 派生（`D2=80,000` 等），与契约默认的 200,000 区分开：
    契约里 200,000 是「未指定时的上限」，实际执行的熔断口径由 D2 决定。
    `task_id` 可用于**恢复**场景：API 层必须复用原始 id（= checkpointer 的
    thread_id），不能每次重新生成。
    """
    from ..contracts import Budget, Payload, Sensitivity, Source, SourceChannel, User, UserRole

    cfg = settings or get_settings()
    now = datetime.now(TZ_CN)
    return TaskEvent(
        task_id=task_id or str(uuid.uuid4()),
        source=Source(channel=SourceChannel.rest_api),
        submitted_at=now,
        user=User(user_id="local_dev", roles=[UserRole.business_user]),
        payload=Payload(text=question),
        sensitivity=Sensitivity.internal,
        budget=Budget(
            max_steps=cfg.default_max_steps,
            max_tokens=cfg.default_max_tokens,
            max_retries=cfg.default_max_retries,
            deadline=now + timedelta(minutes=deadline_minutes),
        ),
    )
