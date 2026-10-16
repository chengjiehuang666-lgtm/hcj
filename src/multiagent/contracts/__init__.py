"""契约层：JSON Schema 的 Pydantic 实现。

每个模块与 `架构设想/接口契约/` 下的同名 Schema 一一对应，
`tests/test_contracts_conformance.py` 会用 Schema 校验本层模型的输出，
保证「代码实现」与「契约」不漂移。

序列化约定：出网（写入黑板、返回调用方）时使用 `to_wire()`，
它会省略值为 None 的可选字段——因为契约中的非 nullable 字段不允许出现 null。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .agent_io import (
    AgentError,
    AgentInvocation,
    AgentResult,
    InvocationBudget,
    ResultStatus,
    TokenUsage,
)
from .blackboard_entry import (
    BlackboardEntry,
    EntryStatus,
    SourceRef,
    SourceType,
    WrittenBy,
)
from .subtask_dag import AgentRole, SubtaskDAG, SubtaskNode
from .task_event import (
    Attachment,
    Budget,
    Intent,
    Payload,
    Priority,
    Sensitivity,
    Source,
    SourceChannel,
    TaskEvent,
    User,
    UserRole,
)

SCHEMA_VERSION = "1.0.0"


def to_wire(model: BaseModel) -> dict[str, Any]:
    """按契约要求序列化：datetime/UUID 转字符串，空的可选字段不输出。"""
    return model.model_dump(mode="json", exclude_none=True)


__all__ = [
    "SCHEMA_VERSION",
    "to_wire",
    # task_event
    "TaskEvent",
    "Source",
    "SourceChannel",
    "User",
    "UserRole",
    "Intent",
    "Attachment",
    "Payload",
    "Sensitivity",
    "Priority",
    "Budget",
    # subtask_dag
    "SubtaskDAG",
    "SubtaskNode",
    "AgentRole",
    # blackboard_entry
    "BlackboardEntry",
    "WrittenBy",
    "EntryStatus",
    "SourceRef",
    "SourceType",
    # agent_io
    "AgentInvocation",
    "AgentResult",
    "ResultStatus",
    "InvocationBudget",
    "TokenUsage",
    "AgentError",
]
