"""需求规划 Agent（Planner）。

职责：把自然语言诉求拆成带依赖的子任务图，并**声明每个节点的输入输出契约与
可判定的验收标准**。产出严格符合 SubtaskDAG 契约，构造期即被校验
（无环、依赖存在、入口可达）。

注意：Planner 是子任务图的**产出者**，自身不是图上的节点。
"""

from __future__ import annotations

import time

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType, SubtaskDAG
from ..gateway import ROUTE_PLANNER
from .base import AgentContext, BaseAgent, _Usage

SYSTEM_PROMPT = """你是任务规划专家，负责把业务分析需求拆解为可由专家 Agent 执行的有向无环任务图。

当前链路只实现了两个可执行节点，**必须严格按此产出**：
1. retriever_1（agent_role=retriever，无依赖）——检索指标口径与数据字典
2. analyst_1（agent_role=analyst，depends_on=["retriever_1"]）——写 SQL 取数并归因

不要新增 reviewer、writer 或任何其它节点：动态执行任意任务图属于后续版本，
现在多规划出来的节点不会被运行。若你认为需要评审环节，请写进 analyst_1 的 acceptance 里。

硬性要求：
1. 节点顺序与依赖必须与上面完全一致，node_id 必须为 retriever_1 与 analyst_1。
2. 每个节点必须有 acceptance 字段，写**可判定的**验收标准（例如「结论中给出至少一个渠道的下滑幅度」），
   禁止写「结果合理」「分析准确」这类无法判定的描述。
3. acceptance 至少 1 条、最多 3 条，宁精勿多。
4. 输出必须是单个 JSON 对象，不要 markdown 围栏，不要任何解释文字。

输出结构（严格遵守字段名）：
{
  "task_id": "<原样回填用户给出的 task_id>",
  "goal": "<对需求的一句话复述>",
  "created_at": "<ISO8601 时间，含时区>",
  "nodes": [
    {"node_id": "retriever_1", "name": "...", "agent_role": "retriever",
     "depends_on": [], "inputs": {}, "outputs": {}, "acceptance": ["..."]},
    {"node_id": "analyst_1", "name": "...", "agent_role": "analyst",
     "depends_on": ["retriever_1"], "inputs": {}, "outputs": {}, "acceptance": ["..."]}
  ]
}"""

# 阶段一 W2 是**固定串行链路**，真正会被执行的只有这两个节点。
# 动态执行任意 DAG 属于 W5。这里对规划结果做硬约束，而不是只靠提示词，
# 避免出现「规划了某节点却从不执行」的静默不一致。
EXECUTABLE_NODES: tuple[str, ...] = ("retriever_1", "analyst_1")


class PlannerAgent(BaseAgent):
    role = "planner"
    version = "0.1.0"

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()

        question = inv.input.get("question") or inv.input.get("text") or ""
        schema = inv.input.get("schema") or ""
        if not question:
            return self.failed(
                inv,
                code="BAD_INPUT",
                message="缺少 input.question",
                retryable=False,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        user = (
            f"task_id: {inv.task_id}\n"
            f"当前时间: {inv.input.get('now', '')}\n"
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

        # 硬约束：规划结果必须落在当前链路能执行的节点集合内。
        # 宁可直接报错，也不要静默地把计划的一部分丢掉。
        unexecutable = [
            node.node_id for node in dag.nodes if node.node_id not in EXECUTABLE_NODES
        ]
        if unexecutable:
            return self.failed(
                inv,
                code="UNSUPPORTED_PLAN",
                message=(
                    f"规划包含当前链路无法执行的节点 {unexecutable}；"
                    f"W2 只支持 {list(EXECUTABLE_NODES)}。"
                    "动态执行任意任务图属于 W5 范围。"
                ),
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
