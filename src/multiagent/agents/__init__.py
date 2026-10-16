"""Agent 层：专家 Agent 的实现，每个都实现 AgentIO 契约。

设计约定：
- Agent 不直接发 HTTP，一律通过 `LLMGateway` / `ModelRouter`（便于降级与成本计量）；
- Agent 不直接连数据库，一律通过 `SqlSandbox`（只读角色 + 语句校验）；
- 失败不抛异常给上层，而是按契约返回 `AgentResult(status=failed,…)`，
  由编排层决定重试、降级还是转人工。
"""

from .analyst import AnalystAgent
from .base import AgentContext, AgentOutputError, BaseAgent
from .planner import PlannerAgent
from .retriever import RetrieverAgent
from .reviewer import ReviewerAgent
from .writer import WriterAgent

__all__ = [
    "BaseAgent",
    "AgentContext",
    "AgentOutputError",
    "PlannerAgent",
    "RetrieverAgent",
    "AnalystAgent",
    "ReviewerAgent",
    "WriterAgent",
    "build_agent_registry",
]


def build_agent_registry() -> dict[str, BaseAgent]:
    """构建 role -> Agent 实例的注册表。

    **这张表同时定义了「Planner 能规划哪些角色」**：图执行前会校验 DAG 里的每个
    `agent_role` 都在这张表里，否则明确报错而不是静默跳过那个节点。

    新增一个专家 Agent 时只需在这里登记——这也是 W6 引入 reviewer/writer 的接入点。

    注意：Planner 也在表中（编排层需要它的实例），但它是**图的产出者**、
    在图之外先运行；传给 Planner 的 `available_roles` 会把它自己排除掉，
    避免它规划出 planner 节点。
    """
    agents: list[BaseAgent] = [
        PlannerAgent(),
        RetrieverAgent(),
        AnalystAgent(),
        ReviewerAgent(),
        WriterAgent(),
    ]
    return {agent.role: agent for agent in agents}
