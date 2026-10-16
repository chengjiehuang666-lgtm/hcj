"""TaskEvent —— 对应 `架构设想/接口契约/task_event.schema.json`。

外部渠道消息归一化后的标准任务事件，是进入编排层的唯一入口结构，
也是全链路追踪主键（task_id）的来源。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _ContractModel(BaseModel):
    """所有契约模型的基类：拒绝未声明字段，与 Schema 的 additionalProperties=false 对齐。"""

    model_config = ConfigDict(extra="forbid")


class SourceChannel(str, Enum):
    rest_api = "rest_api"
    web_console = "web_console"
    im_feishu = "im_feishu"
    im_dingtalk = "im_dingtalk"
    im_wecom = "im_wecom"
    ticket_webhook = "ticket_webhook"


class Sensitivity(str, Enum):
    public = "public"
    internal = "internal"
    confidential = "confidential"
    restricted = "restricted"


class Priority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"


class UserRole(str, Enum):
    business_user = "business_user"
    engineer = "engineer"
    reviewer = "reviewer"
    admin = "admin"


class Source(_ContractModel):
    channel: SourceChannel
    channel_message_id: str | None = None


class User(_ContractModel):
    user_id: str = Field(min_length=1)
    display_name: str | None = None
    roles: list[UserRole] = Field(min_length=1)
    tenant_id: str = "default"


class Intent(_ContractModel):
    category: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    raw_text: str | None = None


class Attachment(_ContractModel):
    name: str
    uri: str
    content_type: str | None = None


class Payload(_ContractModel):
    text: str = Field(min_length=1)
    attachments: list[Attachment] | None = None


class Budget(_ContractModel):
    """执行预算，沿链路逐级传递且只能收紧。"""

    max_steps: int = Field(default=30, ge=1)
    max_tokens: int = Field(default=200_000, ge=1)
    max_retries: int = Field(default=2, ge=0)
    deadline: datetime

    def tightened(self, **overrides: int | datetime) -> Budget:
        """生成一个不宽于自身的预算（用于向子任务下发）。"""
        data = self.model_dump()
        for key, value in overrides.items():
            if key not in data:
                raise ValueError(f"未知预算字段: {key}")
            current = data[key]
            if isinstance(value, int) and isinstance(current, int) and value > current:
                raise ValueError(f"预算只能收紧不能放宽: {key} {current} -> {value}")
            data[key] = value
        return Budget(**data)


class TaskEvent(_ContractModel):
    task_id: UUID
    schema_version: Literal["1.0.0"] = "1.0.0"
    source: Source
    submitted_at: datetime
    user: User
    payload: Payload
    sensitivity: Sensitivity = Sensitivity.internal
    budget: Budget
    intent: Intent | None = None
    priority: Priority = Priority.normal
    context_ref: str | None = None
