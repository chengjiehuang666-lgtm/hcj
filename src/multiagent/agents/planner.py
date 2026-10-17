"""需求规划 Agent（Planner）。

职责：把自然语言诉求拆成带依赖的子任务图，并**声明每个节点的输入输出契约与
可判定的验收标准**。产出严格符合 SubtaskDAG 契约，构造期即被校验
（无环、依赖存在、入口可达）。

注意：Planner 是子任务图的**产出者**，自身不是图上的节点——编排层在构图**之前**
先运行它（图结构依赖它的输出），见 `graph/pipeline.py` 的 `_plan()`。

**W5 变化**：不再限定固定的 node_id。改为按**已注册角色**校验——
Planner 能规划什么，取决于 `build_agent_registry()` 里登记了什么。
"""

from __future__ import annotations

import time

from ..contracts import AgentInvocation, AgentResult, AgentRole, SourceRef, SourceType, SubtaskDAG
from ..gateway import ROUTE_PLANNER
from .base import AgentContext, BaseAgent, _Usage

#: 「高风险动作」的关键词：命中且 DAG 无审批节点 → 拒绝规划（代码门禁）
_HUMAN_APPROVAL_KEYWORDS = (
    "对外发送",
    "发送给",
    "签发",
    "发布",
    "群发",
    "公告",
    "对外",
    "转账",
    "付款",
    "修改数据",
    "删除数据",
)

SYSTEM_PROMPT = """你是任务规划专家，负责把业务分析需求拆解为可由专家 Agent 执行的有向无环任务图。

**可用角色以用户消息里给出的 available_roles 为准，不得规划除此之外的角色。**

各角色的能力与约定：
- retriever：检索。它有三种互不重叠的模式，**必须**用节点自带的 inputs.mode 指定：
    {"mode": "metrics"}   只取指标口径、表结构、数据时间范围（走 SQL）
    {"mode": "knowledge"} 只取业务背景知识：政策变更、历史复盘、数据口径注意事项
    {"mode": "memory"}    只取历史任务的结论（长期记忆）——**仅当问题明确涉及跨会话
                          引用（「上次」「之前」「回顾」「继续」等）时才安排**，
                          与其它检索并行、互不依赖
- analyst：写 SQL 取数并做归因分析、产出结论
- writer：写作。根据上游素材撰写正式文档（报告/简报/通知/方案等），产出 Markdown
- reviewer：评审。逐条核对上游产物的验收标准与证据，识别无依据的结论；
  **它的 depends_on 必须包含图中其余全部节点**（它需要全部材料才能核对），放在链路最后。
- 其它已注册角色：按其命名理解职责

**推荐的默认形态**——两种检索互不依赖，应当并行执行；评审跟在分析之后，依赖全部：
    retriever(mode=metrics)    ─┐
                                 ├─→ analyst_1 ─→ reviewer_1
    retriever(mode=knowledge)  ─┘          ↑
                                          （reviewer_1 依赖前面全部三个节点）
即两个检索节点都**没有依赖**（并行跑），分析节点同时依赖两者（等两路都完成再跑），
评审节点依赖前面所有节点（等材料齐了再逐条核对）。

**文档类需求**（写报告/简报/通知/总结/方案等）用写作形态：
    retriever(mode=knowledge)  ─→ writer_1 ─→ reviewer_1
（需要数据支撑时可再加 retriever(mode=metrics)；writer 与 reviewer 的依赖同样
「要用的材料都列进 depends_on」。）

硬性要求：
1. node_id 用小写字母数字下划线，命名形如 <角色>_<序号>，例如 retriever_1、analyst_1。
2. depends_on 只能引用本图中已存在的 node_id，且不得形成环。
3. 每个节点必须有 acceptance，写**可判定的**验收标准（例如「结论中给出至少一个渠道的
   下滑幅度」），禁止写「结果合理」「分析准确」这类无法判定的描述；至少 1 条、最多 3 条。
   reviewer 节点的 acceptance 写它要核对的具体事项（例如「核对每条结论的数字都有查询
   结果支撑」），**只能引用它依赖的节点会产出的材料**。
4. 每个 retriever 节点必须显式给出 inputs.mode（metrics / knowledge / memory）；
   多个 retriever 节点之间应当互不依赖，以便并行执行；memory 模式只在问题涉及
   历史引用时使用。
5. 分析类需求**必须以 reviewer 节点收尾**，且 reviewer 的 depends_on 列出图中其余
   全部节点；文档类需求必须以 writer 收尾链路（writer → reviewer）收尾；
   节点数量控制在 3-5 个（含 memory 检索时可到 5 个）。
6. 涉及**对外发送、资金/数据变更、文档签发**的节点，必须把
   `requires_human_approval` 置为 true；其余节点写 false（不写按 false 处理）。
7. 输出必须是单个 JSON 对象，不要 markdown 围栏，不要任何解释文字。

输出结构（严格遵守字段名）：
{
  "task_id": "<原样回填用户给出的 task_id>",
  "goal": "<对需求的一句话复述>",
  "created_at": "<ISO8601 时间，含时区>",
  "nodes": [
    {"node_id": "retriever_1", "name": "取指标口径与表结构", "agent_role": "retriever",
     "depends_on": [], "inputs": {"mode": "metrics"}, "outputs": {}, "acceptance": ["..."]},
    {"node_id": "retriever_2", "name": "取业务背景知识", "agent_role": "retriever",
     "depends_on": [], "inputs": {"mode": "knowledge"}, "outputs": {}, "acceptance": ["..."]},
    {"node_id": "analyst_1", "name": "取数并归因", "agent_role": "analyst",
     "depends_on": ["retriever_1", "retriever_2"], "inputs": {}, "outputs": {}, "acceptance": ["..."]},
    {"node_id": "reviewer_1", "name": "交叉校验结论", "agent_role": "reviewer",
     "depends_on": ["retriever_1", "retriever_2", "analyst_1"], "inputs": {}, "outputs": {},
     "acceptance": ["..."], "requires_human_approval": false}
  ]
}"""


class PlannerAgent(BaseAgent):
    role = "planner"
    version = "0.2.0"

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()

        question = inv.input.get("question") or inv.input.get("text") or ""
        schema = inv.input.get("schema") or ""
        available_roles = sorted(inv.input.get("available_roles") or [])
        if not question:
            return self.failed(
                inv,
                code="BAD_INPUT",
                message="缺少 input.question",
                retryable=False,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        if not available_roles:
            # 没有可用角色就无从规划：明确报错，而不是让模型自由发挥
            return self.failed(
                inv,
                code="BAD_INPUT",
                message="缺少 input.available_roles，无法确定可规划的角色",
                retryable=False,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        user = (
            f"task_id: {inv.task_id}\n"
            f"当前时间: {inv.input.get('now', '')}\n"
            f"available_roles: {available_roles}\n\n"
            f"可用数据表:\n{schema}\n\n"
            f"业务需求：{question}"
        )

        try:
            dag = await self.call_json(
                ctx,
                system=SYSTEM_PROMPT,
                user=user,
                model_cls=SubtaskDAG,
                usage=usage,
                route=ROUTE_PLANNER,
            )
        except Exception as exc:
            code, retryable = self.classify_error(exc)
            return self.failed(
                inv,
                code=code,
                message=str(exc)[:500],
                retryable=retryable,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        if reason := self._validate_roles(dag, available_roles):
            return self.failed(
                inv,
                code="UNSUPPORTED_PLAN",
                message=reason,
                retryable=True,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        if reason := self._validate_retriever_modes(dag):
            return self.failed(
                inv,
                code="UNSUPPORTED_PLAN",
                message=reason,
                retryable=True,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        if reason := self._validate_human_approval(dag, question):
            return self.failed(
                inv,
                code="UNSUPPORTED_PLAN",
                message=reason,
                retryable=True,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        if reason := self._validate_reviewer_scope(dag):
            return self.failed(
                inv,
                code="UNSUPPORTED_PLAN",
                message=reason,
                retryable=True,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        return self.ok(
            inv,
            outputs={
                "dag": dag.model_dump(mode="json", exclude_none=True),
                "node_count": len(dag.nodes),
            },
            sources=[
                SourceRef(
                    type=SourceType.model_inference,
                    ref=f"{self.labeled_version}",
                )
            ],
            usage=usage,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    # ------------------------------------------------------------ 规划质量门禁

    @staticmethod
    def _validate_roles(dag: SubtaskDAG, available_roles: list[str]) -> str | None:
        """规划出的角色必须都已注册，否则在**执行前**就报错。

        早报错比晚失败好：等图跑到那个节点才发现没人能执行，已经烧掉了前序节点的 token。
        """
        available = set(available_roles)
        unknown = sorted(
            f"{n.node_id}({n.agent_role.value})"
            for n in dag.nodes
            if n.agent_role.value not in available
        )
        if unknown:
            return (
                f"规划包含未注册角色的节点 {unknown}；可用角色: {sorted(available)}。"
                "请在 agents/build_agent_registry() 中登记后再让 Planner 规划。"
            )
        return None

    @staticmethod
    def _validate_retriever_modes(dag: SubtaskDAG) -> str | None:
        """retriever 的 mode 必须**取值合法**，多个之间必须分工不重复。

        取值校验对准一个真实缺陷（独立复核实跑确认）：非法 mode（如 "policy"）此前
        能通过规划门禁，到执行期才让整个任务以 retriever 的 BAD_INPUT（不可重试）
        失败——规划与全部前序 token 都白烧，违背本文件名下的「早报错比晚失败好」。
        """
        from .retriever import VALID_MODES

        retrievers = [n for n in dag.nodes if n.agent_role is AgentRole.retriever]
        if not retrievers:
            return None
        modes = [str((n.inputs or {}).get("mode") or "").lower() for n in retrievers]
        invalid = [
            f"{n.node_id}(mode={m or '空'})"
            for n, m in zip(retrievers, modes)
            if m and m not in VALID_MODES
        ]
        if invalid:
            return (
                f"retriever 的 inputs.mode 取值非法: {invalid}；"
                f"合法取值为 {sorted(VALID_MODES)}。"
            )
        if len(retrievers) < 2:
            return None
        missing = [n.node_id for n, mode in zip(retrievers, modes) if not mode]
        if missing:
            return (
                f"存在多个 retriever 节点但 {missing} 未声明 inputs.mode；"
                '请用 {"mode": "metrics"} 与 {"mode": "knowledge"} 分工，避免重复检索。'
            )
        duplicated = {m for m in modes if modes.count(m) > 1}
        if duplicated:
            return f"retriever 节点的 inputs.mode 重复: {sorted(duplicated)}；每个模式只应出现一次。"
        return None

    @staticmethod
    def _validate_human_approval(dag: SubtaskDAG, question: str) -> str | None:
        """高风险请求必须规划出审批节点——**代码门禁**，不只靠提示词。

        W6 独立复核的教训：reviewer 依赖完整性只写在提示词里，模型漏了就静默降级。
        这里对「对外发送 / 签发 / 资金或数据变更」类请求做可判定的检查。
        """
        risky = [kw for kw in _HUMAN_APPROVAL_KEYWORDS if kw in question]
        if not risky:
            return None
        if any(n.requires_human_approval for n in dag.nodes):
            return None
        return (
            f"问题含高风险动作（{risky}），但 DAG 没有任何 requires_human_approval=true 的节点；"
            "请把对应节点（如 writer / analyst）标记为需要人工审批。"
        )

    @staticmethod
    def _validate_reviewer_scope(dag: SubtaskDAG) -> str | None:
        """reviewer 的核验范围必须有**代码门禁**，不能只靠提示词。

        独立复核发现：`reviewer 依赖除自己外全部节点` 此前只是提示词约束；漏依赖
        的规划会被照单接受，而条目状态流转与长期记忆沉淀门槛都只看 verdict、
        不看核验范围——未被核对的结论能以「已过评审」身份沉淀入库。

        门禁两条：
        1. **至多一个 reviewer**：多个 reviewer 会各自以乐观锁更新同一批上游条目，
            在同一个 superstep 内并发写同一键，reducer 的「后写覆盖」会静默丢掉
            一方的状态流转（引擎级缺口，属 W7 的冲突消解范围）。当前设计里
            reviewer 是唯一汇点，多 reviewer 的分支形态留待 W7 再开。
        2. reviewer 的 depends_on 必须**覆盖全部非 reviewer 节点**，否则拒绝规划。
        """
        reviewers = [n for n in dag.nodes if n.agent_role is AgentRole.reviewer]
        if not reviewers:
            return None
        if len(reviewers) > 1:
            ids = sorted(n.node_id for n in reviewers)
            return (
                f"规划包含多个 reviewer 节点 {ids}；当前只支持单一汇点评审"
                "（多 reviewer 会并发更新同一批条目，冲突消解属 W7）。"
            )
        reviewer = reviewers[0]
        covered = {n.node_id for n in dag.nodes if n.agent_role is not AgentRole.reviewer}
        missing = sorted(covered - set(reviewer.depends_on))
        if missing:
            return (
                f"reviewer 节点 {reviewer.node_id} 的 depends_on 缺少 {missing}；"
                "评审必须依赖图中其余全部节点，否则未被核对的结论会被标为已审。"
            )
        return None
