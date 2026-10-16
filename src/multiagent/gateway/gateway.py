"""LLM 网关：主备降级与成本记录。

这是 W1 的出口标准之一——「LLM Gateway 打通主备两个模型，完成统一调用与降级」。
"""

from __future__ import annotations

from .base import (
    AllProvidersFailedError,
    CallRecord,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from .openai_compatible import OpenAICompatibleProvider
from .stub import StubProvider


class LLMGateway:
    """按优先级依次尝试供应商，取第一个成功的结果。"""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.history: list[CallRecord] = []

    def providers(self) -> list[LLMProvider]:
        return [p for p in (self.primary, self.fallback) if p is not None]

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        failures: list[tuple[str, str]] = []
        chain: list[str] = []

        for index, provider in enumerate(self.providers()):
            chain.append(provider.name)
            try:
                response = await provider.complete(
                    messages, max_tokens=max_tokens, temperature=temperature
                )
            except Exception as exc:  # 供应商实现各异，统一收敛为降级判断
                # ProviderUnavailableError 自带干净的 reason；其他异常退回 str(exc)，
                # 避免组装出的错误信息里出现重复的供应商名前缀。
                reason = getattr(exc, "reason", None) or str(exc)
                failures.append((provider.name, reason))
                continue

            self.history.append(
                CallRecord(
                    provider=response.provider,
                    model=response.model,
                    ok=True,
                    total_tokens=response.total_tokens,
                    degraded=index > 0,
                    chain=list(chain),
                )
            )
            return response

        self.history.append(
            CallRecord(provider="", model="", ok=False, degraded=True, chain=list(chain))
        )
        raise AllProvidersFailedError(failures)


def build_gateway(
    *,
    mode: str,
    primary_provider: str,
    primary_model: str,
    primary_base_url: str,
    primary_api_key: str | None,
    fallback_provider: str,
    fallback_model: str,
    fallback_base_url: str,
    fallback_api_key: str | None,
) -> LLMGateway:
    """按模式装配网关。

    stub 模式：两个 StubProvider，不需要密钥，用于本地跑通链路。
    live 模式：两个 OpenAI 兼容供应商，主备降级。
    """
    if mode == "stub":
        return LLMGateway(
            primary=StubProvider(name=f"stub:{primary_provider}", model=primary_model),
            fallback=StubProvider(name=f"stub:{fallback_provider}", model=fallback_model),
        )

    return LLMGateway(
        primary=OpenAICompatibleProvider(
            provider=primary_provider,
            model=primary_model,
            base_url=primary_base_url,
            api_key=primary_api_key,
        ),
        fallback=OpenAICompatibleProvider(
            provider=fallback_provider,
            model=fallback_model,
            base_url=fallback_base_url,
            api_key=fallback_api_key,
        ),
    )


def build_gateway_from_settings(settings) -> LLMGateway:
    """从 Settings 装配网关（生产代码走这个入口）。"""
    return build_gateway(
        mode=settings.llm_mode,
        primary_provider=settings.primary.provider,
        primary_model=settings.primary.model,
        primary_base_url=settings.primary.base_url,
        primary_api_key=(
            settings.primary.api_key.get_secret_value() if settings.primary.api_key else None
        ),
        fallback_provider=settings.fallback.provider,
        fallback_model=settings.fallback.model,
        fallback_base_url=settings.fallback.base_url,
        fallback_api_key=(
            settings.fallback.api_key.get_secret_value() if settings.fallback.api_key else None
        ),
    )


__all__ = ["LLMGateway", "build_gateway", "build_gateway_from_settings"]
