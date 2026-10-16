"""LLM 网关：供应商无关的统一调用与降级。

设计要点：
- 业务代码只依赖 `LLMProvider` 抽象，不直接发 HTTP；换供应商只改配置。
- DeepSeek / 通义千问 Qwen / 智谱 GLM 均提供 OpenAI 兼容接口，
  因此三者共用 `OpenAICompatibleProvider` 一个实现。
- `LLMGateway` 负责主备降级：主模型失败或未配置 key 时自动切备用，
  两次都失败才抛错，并在异常中保留两条失败原因，避免静默失败。
"""

from .base import (
    AllProvidersFailedError,
    GatewayError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    MessageRole,
    ProviderUnavailableError,
)
from .gateway import LLMGateway, build_gateway, build_gateway_from_settings
from .openai_compatible import OpenAICompatibleProvider
from .stub import StubProvider

__all__ = [
    "LLMGateway",
    "build_gateway",
    "build_gateway_from_settings",
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "MessageRole",
    "GatewayError",
    "ProviderUnavailableError",
    "AllProvidersFailedError",
    "StubProvider",
    "OpenAICompatibleProvider",
]
