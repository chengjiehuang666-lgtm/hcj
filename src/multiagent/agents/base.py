"""Agent 基类与共享工具。

关键点：**结构化输出的解析必须有容错与重试**。模型偶尔会返回带 markdown 围栏的
JSON、或字段名写错，直接 `json.loads` 会让整条链路随机失败。这里统一用
「提取 JSON → Pydantic 校验 → 失败则把错误回灌给模型再试一次」的策略。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..contracts import (
    AgentError,
    AgentInvocation,
    AgentResult,
    ResultStatus,
    SourceRef,
    SourceType,
    TokenUsage,
)
from ..gateway import (
    ROUTE_DEFAULT,
    AllProvidersFailedError,
    EmptyCompletionError,
    GatewayError,
    LLMGateway,
    LLMMessage,
    ModelRouter,
)
from ..tools import SqlExecutionError, SqlSandbox

TModel = TypeVar("TModel", bound=BaseModel)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class AgentOutputError(Exception):
    """模型输出无法解析为期望的结构。"""


@dataclass
class AgentContext:
    """一次任务执行期间 Agent 可用的资源。

    `gateway` 既可以是 `LLMGateway`（单条主备链），也可以是 `ModelRouter`
    （分级路由）。两者具有完全相同的 `complete()` 签名，因此可互换；
    传 `ModelRouter` 时，调用方通过 `route` 指定调用点来选档位。

    `registry` 是 role -> Agent 实例的映射（见 `build_agent_registry`），
    编排层用它把 DAG 节点分派给对应 Agent，并校验 DAG 里的角色确实可执行。
    """

    settings: Settings
    gateway: LLMGateway | ModelRouter
    sandbox: SqlSandbox
    #: role -> Agent 实例。留空时表示"不走注册表"（单测构造 ctx 时常见）
    registry: dict[str, BaseAgent] = field(default_factory=dict)


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, response) -> None:
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.reasoning_tokens += response.reasoning_tokens

    def to_contract(self) -> TokenUsage:
        return TokenUsage(
            input=self.input_tokens,
            output=self.output_tokens,
            total=self.input_tokens + self.output_tokens,
        )


def extract_json(text: str) -> Any:
    """从模型输出里提取 JSON，容忍 markdown 围栏与前后废话。"""
    candidate = text.strip()
    if fenced := _JSON_FENCE.search(candidate):
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 退一步：截取第一个 { 到最后一个 }（或 [ ]）
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise AgentOutputError(f"模型输出不是合法 JSON：{text[:200]!r}")


class BaseAgent(ABC):
    """专家 Agent 基类。"""

    role: str = "base"
    version: str = "0.1.0"

    @property
    def labeled_version(self) -> str:
        """写入黑板与结果的身份标识，用于「责任 Agent」可追溯。"""
        return f"{self.role}@{self.version}"

    # ---- 模型调用 ----
    async def call_json(
        self,
        ctx: AgentContext,
        *,
        system: str,
        user: str,
        model_cls: type[TModel],
        usage: _Usage,
        max_tokens: int | None = None,
        attempts: int = 2,
        route: str = ROUTE_DEFAULT,
    ) -> TModel:
        """要求模型输出符合 `model_cls` 的 JSON，解析失败时把错误回灌重试。

        `route` 标识**调用点**（见 `gateway/router.py` 的 `ROUTE_*`）。
        当 `ctx.gateway` 是 `ModelRouter` 时，它据此选择档位——例如把「生成 SQL」
        这类机械步骤路由到 fast 档的低延迟模型，把归因留给推理型模型。
        """
        if max_tokens is None:
            max_tokens = ctx.settings.llm_max_output_tokens
        messages = [LLMMessage.system(system), LLMMessage.user(user)]
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            response = await ctx.gateway.complete(
                messages, max_tokens=max_tokens, route=route
            )
            usage.add(response)
            try:
                payload = extract_json(response.content)
                return model_cls.model_validate(payload)
            except (AgentOutputError, ValidationError) as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                messages = [
                    LLMMessage.system(system),
                    LLMMessage.user(user),
                    LLMMessage.assistant(response.content[:1500]),
                    LLMMessage.user(
                        "上面的输出无法通过校验，错误如下：\n"
                        f"{str(exc)[:600]}\n\n"
                        "请只输出修正后的 JSON，不要任何解释文字或 markdown 围栏。"
                    ),
                ]

        raise AgentOutputError(f"连续 {attempts} 次未能得到合法输出：{last_error}")

    # ---- 结果构造 ----
    def ok(
        self,
        inv: AgentInvocation,
        *,
        outputs: dict[str, Any],
        sources: list[SourceRef],
        usage: _Usage,
        duration_ms: int,
        confidence: float | None = None,
        blackboard_writes: list[str] | None = None,
    ) -> AgentResult:
        return AgentResult(
            task_id=inv.task_id,
            node_id=inv.node_id,
            attempt=inv.attempt,
            status=ResultStatus.ok,
            tokens_used=usage.to_contract(),
            duration_ms=duration_ms,
            outputs=outputs,
            confidence=confidence,
            sources=sources,
            blackboard_writes=blackboard_writes or [],
        )

    def failed(
        self,
        inv: AgentInvocation,
        *,
        code: str,
        message: str,
        retryable: bool,
        usage: _Usage,
        duration_ms: int,
    ) -> AgentResult:
        return AgentResult(
            task_id=inv.task_id,
            node_id=inv.node_id,
            attempt=inv.attempt,
            status=ResultStatus.failed,
            tokens_used=usage.to_contract(),
            duration_ms=duration_ms,
            error=AgentError(code=code, message=message, retryable=retryable),
        )

    @staticmethod
    def classify_error(exc: Exception) -> tuple[str, bool]:
        """把底层异常映射为契约里的稳定错误码 + 是否可重试。"""
        if isinstance(exc, EmptyCompletionError):
            return "EMPTY_COMPLETION", True
        if isinstance(exc, AllProvidersFailedError):
            return "PROVIDER_UNAVAILABLE", True
        if isinstance(exc, GatewayError):
            return "GATEWAY_ERROR", True
        if isinstance(exc, SqlExecutionError):
            return "SQL_ERROR", True
        if isinstance(exc, AgentOutputError):
            return "SCHEMA_MISMATCH", True
        return "INTERNAL_ERROR", False

    # ---- 子类实现 ----
    @abstractmethod
    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        """执行一次 Agent 调用，返回符合 AgentIO 契约的结果。"""
