"""LLM 网关：主备降级与成本记录。

这是 W1 的出口标准之一——「LLM Gateway 打通主备两个模型，完成统一调用与降级」。

本类只管**一条**供应商链（主 + 备）。跨档位的选择由 `router.ModelRouter` 负责，
它按调用点把请求分派到不同的 LLMGateway 实例上。
"""

from __future__ import annotations

import time

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
        route: str | None = None,
    ) -> LLMResponse:
        """按优先级依次尝试供应商，取第一个成功的结果。

        `route` 只用于可观测性（原样写入 CallRecord）：本类不决定路由，
        选档位是 `ModelRouter` 的职责。保留该参数是为了让 LLMGateway 与
        ModelRouter 具有**完全相同的调用签名**，从而可以直接互换。
        """
        failures: list[tuple[str, str]] = []
        chain: list[str] = []
        started = time.perf_counter()

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
                    route=route,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            )
            return response

        self.history.append(
            CallRecord(
                provider="",
                model="",
                ok=False,
                degraded=True,
                chain=list(chain),
                route=route,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
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


def build_gateway_from_providers(mode: str, primary, fallback) -> LLMGateway:
    """从两个 ProviderConfig 装配**一条**主备链。

    分级路由需要按档位装配多条链，故把装配逻辑抽到这里复用。
    """

    def _secret(config) -> str | None:
        return config.api_key.get_secret_value() if config.api_key else None

    return build_gateway(
        mode=mode,
        primary_provider=primary.provider,
        primary_model=primary.model,
        primary_base_url=primary.base_url,
        primary_api_key=_secret(primary),
        fallback_provider=fallback.provider,
        fallback_model=fallback.model,
        fallback_base_url=fallback.base_url,
        fallback_api_key=_secret(fallback),
    )


def build_gateway_from_settings(settings) -> LLMGateway:
    """从 Settings 的默认（reasoning）档装配单条主备链。

    生产代码请用 `router.build_router_from_settings()`（支持分级路由）；
    本函数保留给只需单条链的场景与既有测试。
    """
    return build_gateway_from_providers(settings.llm_mode, settings.primary, settings.fallback)


__all__ = [
    "LLMGateway",
    "build_gateway",
    "build_gateway_from_providers",
    "build_gateway_from_settings",
]
