"""LLM 网关的抽象与数据结构，不含具体供应商实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: MessageRole
    content: str

    @staticmethod
    def system(content: str) -> LLMMessage:
        return LLMMessage(MessageRole.system, content)

    @staticmethod
    def user(content: str) -> LLMMessage:
        return LLMMessage(MessageRole.user, content)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GatewayError(Exception):
    """网关层错误基类。"""


class ProviderUnavailableError(GatewayError):
    """单个供应商不可用（未配置密钥、网络失败、返回异常）。"""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"供应商 '{provider}' 不可用: {reason}")


class AllProvidersFailedError(GatewayError):
    """主备供应商全部失败。保留全部原因，避免静默失败。"""

    def __init__(self, failures: list[tuple[str, str]]) -> None:
        self.failures = failures
        detail = "; ".join(f"{p}: {r}" for p, r in failures)
        super().__init__(f"全部供应商调用失败 -> {detail}")


class LLMProvider(ABC):
    """供应商无关的模型调用抽象。"""

    name: str
    model: str

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """单轮补全。失败时抛 ProviderUnavailableError。"""

    @abstractmethod
    async def health(self) -> bool:
        """轻量可用性探测，用于降级决策。"""


@dataclass
class CallRecord:
    """一次网关调用的记录，供成本计量（W10）与审计使用。"""

    provider: str
    model: str
    ok: bool
    total_tokens: int = 0
    degraded: bool = False
    chain: list[str] = field(default_factory=list)
