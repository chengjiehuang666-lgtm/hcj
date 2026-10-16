"""AgentIO —— 对应 `架构设想/接口契约/agent_io.schema.json`。

统一 Agent 调用契约：Orchestrator 用 `AgentInvocation` 调用任意专家 Agent，
Agent 用 `AgentResult` 返回。与具体角色无关——角色差异体现在 SubtaskDAG
节点的 inputs/outputs 上，而非本契约。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .blackboard_entry import SourceRef


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResultStatus(str, Enum):
    ok = "ok"
    failed = "failed"
    needs_human = "needs_human"
    needs_replan = "needs_replan"


class InvocationBudget(_ContractModel):
    max_tokens: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)


class TokenUsage(_ContractModel):
    input: int = Field(ge=0)
    output: int = Field(ge=0)
    total: int = Field(ge=0)


class AgentError(_ContractModel):
    code: str = Field(
        description="稳定错误码（机器可判别），如 SCHEMA_MISMATCH / SQL_ERROR / TIMEOUT / BUDGET_EXCEEDED"
    )
    message: str
    retryable: bool


class AgentInvocation(_ContractModel):
    task_id: str
    schema_version: Literal["1.0.0"] = "1.0.0"
    node_id: str
    attempt: int = Field(ge=1, description="与 task_id、node_id 共同构成幂等键")
    agent_role: str
    trace_id: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    budget: InvocationBudget
    visible_blackboard_keys: list[str] = Field(default_factory=list)


class AgentResult(_ContractModel):
    task_id: str
    schema_version: Literal["1.0.0"] = "1.0.0"
    node_id: str
    attempt: int = Field(ge=1)
    status: ResultStatus
    tokens_used: TokenUsage
    duration_ms: int = Field(ge=0)
    outputs: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    sources: list[SourceRef] | None = Field(default=None, min_length=1)
    error: AgentError | None = None
    blackboard_writes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_status_requirements(self) -> AgentResult:
        """契约中由 allOf/if-then 表达的条件必填，在此以校验器落实。"""
        if self.status is ResultStatus.ok:
            if self.outputs is None:
                raise ValueError("status=ok 时 outputs 必填")
            if not self.sources:
                raise ValueError(
                    "status=ok 时 sources 必填且不得为空——"
                    "契约要求每个结论可回溯到来源"
                )
        if self.status is ResultStatus.failed and self.error is None:
            raise ValueError("status=failed 时 error 必填")
        return self
