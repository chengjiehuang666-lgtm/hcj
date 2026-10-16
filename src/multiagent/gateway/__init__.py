"""LLM 网关：供应商无关的统一调用、主备降级与分级路由。

- `LLMGateway`：一条主备供应商链（主失败自动切备）
- `ModelRouter`：按调用点把请求分派到不同档位的 LLMGateway（分级路由）
- 两者 `complete()` 签名一致，可直接互换；业务侧只依赖这个签名。
"""

from .base import (
    AllProvidersFailedError,
    CallRecord,
    EmptyCompletionError,
    GatewayError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    MessageRole,
    ProviderUnavailableError,
)
from .gateway import (
    LLMGateway,
    build_gateway,
    build_gateway_from_providers,
    build_gateway_from_settings,
)
from .openai_compatible import OpenAICompatibleProvider
from .router import (
    KNOWN_ROUTES,
    ROUTE_ANALYST_CONCLUDE,
    ROUTE_ANALYST_SQL,
    ROUTE_DEFAULT,
    ROUTE_PLANNER,
    ROUTE_REVIEWER,
    ROUTE_WRITER,
    TIER_FAST,
    TIER_REASONING,
    ModelRouter,
    build_router_from_settings,
)
from .stub import StubProvider

__all__ = [
    # 网关与路由器
    "LLMGateway",
    "ModelRouter",
    "build_gateway",
    "build_gateway_from_providers",
    "build_gateway_from_settings",
    "build_router_from_settings",
    # 档位与调用点
    "TIER_REASONING",
    "TIER_FAST",
    "ROUTE_PLANNER",
    "ROUTE_ANALYST_SQL",
    "ROUTE_ANALYST_CONCLUDE",
    "ROUTE_REVIEWER",
    "ROUTE_WRITER",
    "ROUTE_DEFAULT",
    "KNOWN_ROUTES",
    # 数据结构与异常
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "MessageRole",
    "CallRecord",
    "GatewayError",
    "ProviderUnavailableError",
    "EmptyCompletionError",
    "AllProvidersFailedError",
    # 具体实现
    "StubProvider",
    "OpenAICompatibleProvider",
]
