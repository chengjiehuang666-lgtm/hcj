"""Agent 层：专家 Agent 的实现，每个都实现 AgentIO 契约。

设计约定：
- Agent 不直接发 HTTP，一律通过 `LLMGateway`（便于降级与成本计量）；
- Agent 不直接连数据库，一律通过 `SqlSandbox`（只读角色 + 语句校验）；
- 失败不抛异常给上层，而是按契约返回 `AgentResult(status=failed,…)`，
  由 Orchestrator 决定重试、降级还是转人工。
"""

from .analyst import AnalystAgent
from .base import AgentContext, AgentOutputError, BaseAgent
from .planner import PlannerAgent
from .retriever import RetrieverAgent

__all__ = [
    "BaseAgent",
    "AgentContext",
    "AgentOutputError",
    "PlannerAgent",
    "RetrieverAgent",
    "AnalystAgent",
]
