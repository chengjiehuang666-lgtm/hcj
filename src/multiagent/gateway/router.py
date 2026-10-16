"""分级路由：按「调用点」把请求分派到不同档位的模型。

**为什么需要它**：20 条评测基线显示 P95 端到端时延 133.8 秒（M1 目标 ≤ 60 秒），
根因不是编排逻辑，而是两个可用端点都是**推理型模型**——连「生成一条 SQL」这种
机械步骤也要烧 2,000–4,000 个思考 token，单次调用就可能 40–80 秒。
《系统架构设计》2.2.4 早已给出对策：

    分级路由：简单分类/抽取走小模型（低成本低延迟），复杂规划/推理走强推理模型。

**为什么按调用点而不是只按 Agent 角色**：Analyst 的两次调用性质完全不同——
「生成 SQL」是机械翻译，「归因」是真推理。这正是最该区分的地方。

**档位**：
- `reasoning`：复杂规划与归因，用强推理模型（对应原有 `LLM_PRIMARY_*` / `LLM_FALLBACK_*`）
- `fast`：机械步骤（生成 SQL、抽取、分类），用低延迟小模型（`LLM_FAST_*`）

**路由表来自配置**（`LLM_ROUTING`，JSON），改路由不需要改代码：

    LLM_ROUTING={"planner":"reasoning","analyst_sql":"fast","analyst_conclude":"reasoning"}
"""

from __future__ import annotations

from .base import (
    AllProvidersFailedError,
    CallRecord,
    EmptyCompletionError,
    LLMMessage,
    LLMResponse,
)
from .gateway import LLMGateway, build_gateway_from_providers

# ---------------------------------------------------------------- 档位名

TIER_REASONING = "reasoning"
TIER_FAST = "fast"

# ---------------------------------------------------------------- 调用点

ROUTE_PLANNER = "planner"
ROUTE_ANALYST_SQL = "analyst_sql"
ROUTE_ANALYST_CONCLUDE = "analyst_conclude"
ROUTE_REVIEWER = "reviewer"
ROUTE_WRITER = "writer"
ROUTE_DEFAULT = "default"

#: 代码中真实存在的调用点。新建调用点时请一并登记，便于发现漏配。
KNOWN_ROUTES: tuple[str, ...] = (
    ROUTE_PLANNER,
    ROUTE_ANALYST_SQL,
    ROUTE_ANALYST_CONCLUDE,
    ROUTE_REVIEWER,
    ROUTE_WRITER,
)


#: 输出上限被思考吃满而截断时，放宽重试用到的上限。
DEFAULT_ESCALATION_TOKENS = 8192


class ModelRouter:
    """按 `route` 把调用分派到对应档位的 `LLMGateway`。

    与 `LLMGateway` 具有**完全相同的 `complete()` 签名**，因此两者可直接互换——
    这也是业务侧（`agents/base.py`）只需一处改动的原因。
    """

    def __init__(
        self,
        tiers: dict[str, LLMGateway],
        routing: dict[str, str] | None = None,
        *,
        default_tier: str = TIER_REASONING,
        output_caps: dict[str, int] | None = None,
    ) -> None:
        if not tiers:
            raise ValueError("至少需要一个档位")
        self.tiers = dict(tiers)
        self.routing = dict(routing or {})
        self.default_tier = default_tier if default_tier in self.tiers else next(iter(self.tiers))
        #: 档位 -> 单次调用输出上限。**这是 fast 档延迟的主要控制点**：
        #: 实测给本地模型 8192 的上限时它会思考 453 token、耗时 10.7 秒；
        #: 收紧到 2048 后思考归零、耗时降到 4.5 秒（输入完全相同）。
        #: 过大的上限等于邀请模型无限思考。
        self.output_caps = dict(output_caps or {})
        #: 按发生顺序记录每一次调用（跨档位），供成本与时延归因
        self.history: list[CallRecord] = []
        self._unmapped_routes: set[str] = set()

    # ---- 路由决策 ----
    def tier_for(self, route: str | None) -> str:
        if route is None:
            return self.default_tier
        tier = self.routing.get(route)
        if tier is None:
            # 未配置的调用点退回默认档，并记账——出现即说明路由表漏配
            self._unmapped_routes.add(route)
            return self.default_tier
        if tier not in self.tiers:
            raise KeyError(
                f"路由 {route!r} 指向不存在的档位 {tier!r}；可用档位: {sorted(self.tiers)}"
            )
        return tier

    def gateway_for(self, route: str | None) -> LLMGateway:
        return self.tiers[self.tier_for(route)]

    def output_cap_for(self, route: str | None) -> int | None:
        """该调用点所属档位的输出上限；未配置则为 None（不钳制）。"""
        return self.output_caps.get(self.tier_for(route))

    @property
    def unmapped_routes(self) -> set[str]:
        """被调用过但未在路由表中配置的调用点。非空说明路由表漏配。"""
        return set(self._unmapped_routes)

    # ---- 与 LLMGateway 同签名 ----
    async def _dispatch(
        self,
        gateway: LLMGateway,
        messages: list[LLMMessage],
        max_tokens: int | None,
        temperature: float | None,
        route: str | None,
    ) -> LLMResponse:
        """调用指定档位并汇入总账（成功失败都记）。"""
        before = len(gateway.history)
        try:
            return await gateway.complete(
                messages, max_tokens=max_tokens, temperature=temperature, route=route
            )
        finally:
            self.history.extend(gateway.history[before:])

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        route: str | None = None,
    ) -> LLMResponse:
        # 按档位钳制输出上限：调用方只能要求「不多于」本档上限，不能放宽。
        # 目的是把「模型思考失控」这个失效模式掐在网关层，而不是靠每个调用点自觉。
        cap = self.output_cap_for(route)
        requested = max_tokens
        effective = requested
        if cap is not None and (effective is None or effective > cap):
            effective = cap

        gateway = self.gateway_for(route)
        try:
            return await self._dispatch(gateway, messages, effective, temperature, route)
        except AllProvidersFailedError as exc:
            # 收紧上限会引入一个新失效模式：模型一旦决定深入思考，会把上限全烧在
            # 思考上、正文为空（实测 2048 全部被思考消耗）。这类失败**不是能力问题
            # 而是额度问题**，放宽一次即可恢复；其余失败照常抛出。
            if not _all_truncated(exc):
                raise
            relaxed = requested if requested is not None else DEFAULT_ESCALATION_TOKENS
            if effective is not None and relaxed <= effective:
                raise
            return await self._dispatch(gateway, messages, relaxed, temperature, route)

    # ---- 归因辅助 ----
    def summarize_by_route(self) -> dict[str, dict[str, int]]:
        """按调用点汇总调用次数、token 与耗时，用于分级路由的归因分析。"""
        summary: dict[str, dict[str, int]] = {}
        for record in self.history:
            key = record.route or "(未标注)"
            bucket = summary.setdefault(
                key,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_ms": 0,
                },
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += record.input_tokens
            bucket["output_tokens"] += record.output_tokens
            bucket["total_tokens"] += record.total_tokens
            bucket["duration_ms"] += record.duration_ms
        return summary


def _all_truncated(exc: AllProvidersFailedError) -> bool:
    """判断整条链是否都因「输出上限被思考吃满」而失败。

    只看失败原因文本不可靠，因此 AllProvidersFailedError 会保留原始异常；
    这里优先用异常类型判断，退化时才回退到文本匹配。
    """
    causes = getattr(exc, "causes", None)
    if causes:
        return all(
            isinstance(c, EmptyCompletionError) and c.finish_reason == "length"
            for c in causes
        )
    # 文本回退：EmptyCompletionError 的 message 固定含 finish_reason=length
    return bool(exc.failures) and all(
        "finish_reason=length" in reason for _, reason in exc.failures
    )


def build_router_from_settings(settings) -> ModelRouter:
    """按配置装配路由器。stub 模式下所有档位都由 StubProvider 承担。"""
    tiers = {
        TIER_REASONING: build_gateway_from_providers(
            settings.llm_mode, settings.primary, settings.fallback
        ),
        TIER_FAST: build_gateway_from_providers(
            settings.llm_mode, settings.fast_primary, settings.fast_fallback
        ),
    }
    return ModelRouter(
        tiers,
        getattr(settings, "llm_routing", None),
        default_tier=getattr(settings, "llm_default_tier", TIER_REASONING),
        output_caps={
            TIER_REASONING: getattr(settings, "llm_max_output_tokens", 8192),
            TIER_FAST: getattr(settings, "llm_fast_max_output_tokens", 2048),
        },
    )


__all__ = [
    "ModelRouter",
    "build_router_from_settings",
    "TIER_REASONING",
    "TIER_FAST",
    "ROUTE_PLANNER",
    "ROUTE_ANALYST_SQL",
    "ROUTE_ANALYST_CONCLUDE",
    "ROUTE_DEFAULT",
    "KNOWN_ROUTES",
]
