"""BlackboardEntry —— 对应 `架构设想/接口契约/blackboard_entry.schema.json`。

黑板条目：Agent 之间不直接传递私有状态，一律通过黑板交换，
黑板是单次任务内的唯一事实源。写入采用条目级版本号乐观锁。

`sources` 不得为空——这是产品需求「每个结论可回溯到来源数据、调用工具、
模型版本与责任 Agent」在契约层的强制落点。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntryStatus(str, Enum):
    draft = "draft"
    validated = "validated"
    rejected = "rejected"
    superseded = "superseded"


class SourceType(str, Enum):
    sql = "sql"
    document = "document"
    web = "web"
    model_inference = "model_inference"
    human = "human"
    tool = "tool"


class WrittenBy(_ContractModel):
    """写入者身份，是「责任 Agent」可追溯性的落点。"""

    agent_role: str
    agent_version: str = Field(min_length=1)
    node_id: str
    attempt: int = Field(ge=1)


class SourceRef(_ContractModel):
    """来源引用。agent_io 的 sources 复用本结构。"""

    type: SourceType
    ref: str = Field(min_length=1)
    timestamp: datetime | None = None
    retrieval_score: float | None = None


class BlackboardEntry(_ContractModel):
    task_id: str
    schema_version: Literal["1.0.0"] = "1.0.0"
    key: str = Field(
        pattern=r"^[a-z0-9_]+\.[a-z0-9_]+$",
        description="命名约定 <node_id>.<output_name>，与 SubtaskDAG 节点 outputs 字段名对应",
    )
    version: int = Field(ge=1)
    payload: Any
    written_by: WrittenBy
    written_at: datetime
    status: EntryStatus
    sources: list[SourceRef] = Field(min_length=1)
    superseded_by: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    ttl_seconds: int | None = Field(default=None, ge=1)

    def next_version(self, payload: Any, **updates: Any) -> BlackboardEntry:
        """基于当前条目生成下一个版本（乐观锁：调用方须先读到本条目）。"""
        return self.model_copy(
            update={
                "payload": payload,
                "version": self.version + 1,
                "superseded_by": None,
                **updates,
            }
        )
