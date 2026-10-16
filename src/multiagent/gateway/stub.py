"""Stub 供应商：不需要任何 API key 的假实现。

用途：
1. 本地在 `LLM_MODE=stub` 下跑通完整链路，不被密钥与网络阻塞；
2. 测试中模拟「主供应商失败 → 降级到备用」，验证网关降级逻辑。

响应是确定性的（同样的输入得到同样的输出），便于断言。
"""

from __future__ import annotations

from .base import LLMMessage, LLMProvider, LLMResponse, ProviderUnavailableError

# 粗略估算：约 4 个字符 1 个 token。仅用于本地计量演示，不用于真实计费。
_CHARS_PER_TOKEN = 4


class StubProvider(LLMProvider):
    def __init__(
        self,
        name: str = "stub",
        model: str = "stub-model",
        *,
        response_text: str = "[stub] 这是假的模型响应，仅用于跑通链路。",
        should_fail: bool = False,
        fail_reason: str = "stub 被配置为失败（用于测试降级）",
    ) -> None:
        self.name = name
        self.model = model
        self._response_text = response_text
        self._should_fail = should_fail
        self._fail_reason = fail_reason

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if self._should_fail:
            raise ProviderUnavailableError(self.name, self._fail_reason)

        # 确定性响应：把最后一条用户消息的前若干字符回显，便于确认路由正确
        last_user = next(
            (m.content for m in reversed(messages) if m.role.value == "user"), ""
        )
        echo = last_user[:40]
        content = f"{self._response_text} [echo={echo}]" if echo else self._response_text

        input_chars = sum(len(m.content) for m in messages)
        return LLMResponse(
            content=content,
            provider=self.name,
            model=self.model,
            input_tokens=max(1, input_chars // _CHARS_PER_TOKEN),
            output_tokens=max(1, len(content) // _CHARS_PER_TOKEN),
            finish_reason="stop",
        )

    async def health(self) -> bool:
        return not self._should_fail
