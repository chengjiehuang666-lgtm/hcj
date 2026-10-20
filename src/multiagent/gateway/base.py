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

    @staticmethod
    def assistant(content: str) -> LLMMessage:
        """用于把模型的上一轮输出回灌给它（结构化输出的纠错重试需要）。"""
        return LLMMessage(MessageRole.assistant, content)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def truncated(self) -> bool:
        """是否因 token 额度耗尽被截断（finish_reason=length）。"""
        return self.finish_reason == "length"


class GatewayError(Exception):
    """网关层错误基类。"""


class ProviderUnavailableError(GatewayError):
    """单个供应商不可用（未配置密钥、网络失败、返回异常）。"""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"供应商 '{provider}' 不可用: {reason}")


class EmptyCompletionError(GatewayError):
    """模型返回了空内容。

    对**推理型模型**（如 deepseek-flash、GLM-Z 系列）这是常见陷阱：
    它们先产生 reasoning_content，再产生 content，而 max_tokens 由两者共享。
    额度偏小时，全部 token 会被思考过程吃掉，content 为空且 finish_reason=length。
    这类结果必须显式报错，绝不能当作正常的空字符串返回给调用方。
    """

    def __init__(self, provider: str, finish_reason: str | None, reasoning_tokens: int) -> None:
        self.provider = provider
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens
        hint = (
            f"（finish_reason={finish_reason}，思考已消耗 {reasoning_tokens} tokens，"
            "推理型模型需显著调大 max_tokens）"
            if reasoning_tokens
            else f"（finish_reason={finish_reason}）"
        )
        self.reason = f"模型返回空内容 {hint}"
        super().__init__(f"供应商 '{provider}' 返回空内容 {hint}")


class AllProvidersFailedError(GatewayError):
    """主备供应商全部失败。保留全部原因，避免静默失败。"""

    def __init__(
        self,
        failures: list[tuple[str, str]],
        causes: list[BaseException] | None = None,
    ) -> None:
        self.failures = failures
        #: 原始异常对象。供上层做**类型判断**（例如区分「额度被思考吃满而截断」
        #: 与「供应商真的挂了」），比匹配错误文本可靠。
        self.causes: list[BaseException] = list(causes or [])
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
    """一次网关调用的记录，供成本计量（W10）、分级路由归因与审计使用。"""

    provider: str
    model: str
    ok: bool
    total_tokens: int = 0
    degraded: bool = False
    chain: list[str] = field(default_factory=list)
    # ---- 分级路由归因用 ----
    # route：标识本次调用来自哪个调用点（planner / analyst_sql / …），
    #        直接使用 LLMGateway 时为 None。
    # duration_ms：本次调用（含降级尝试）的墙钟耗时。
    route: str | None = None
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # ---- W10：task_id 贯穿与按 Agent 归因 ----
    task_id: str | None = None
    node_id: str | None = None
    agent_role: str | None = None
