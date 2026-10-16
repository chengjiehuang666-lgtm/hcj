"""分级路由测试。

路由是「用便宜模型跑机械步骤、用推理模型跑真推理」这条成本/时延策略的实现，
它必须：(1) 按调用点分派到正确档位；(2) 漏配时退回默认档且可被发现；
(3) 与 LLMGateway 保持同签名以保证可互换；(4) 逐次明细可用于归因。
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.config import Settings  # noqa: E402
from multiagent.gateway import (  # noqa: E402
    ROUTE_ANALYST_SQL,
    ROUTE_DEFAULT,
    ROUTE_PLANNER,
    TIER_FAST,
    TIER_REASONING,
    AllProvidersFailedError,
    EmptyCompletionError,
    LLMGateway,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ModelRouter,
    ProviderUnavailableError,
    StubProvider,
    build_gateway_from_providers,
    build_router_from_settings,
)

settings = Settings(_env_file=ROOT / ".env")


def _gateway(name: str) -> LLMGateway:
    return LLMGateway(primary=StubProvider(name=name, model=f"{name}-model"))


def _router(routing: dict[str, str] | None = None) -> ModelRouter:
    return ModelRouter(
        tiers={TIER_REASONING: _gateway("reasoning"), TIER_FAST: _gateway("fast")},
        routing=routing or {},
    )


# ------------------------------------------------------------------ 路由决策


def test_routes_configured_call_site_to_its_tier() -> None:
    router = _router({ROUTE_ANALYST_SQL: TIER_FAST, ROUTE_PLANNER: TIER_REASONING})
    assert router.tier_for(ROUTE_ANALYST_SQL) == TIER_FAST
    assert router.tier_for(ROUTE_PLANNER) == TIER_REASONING


def test_unconfigured_call_site_falls_back_to_default_and_is_recorded() -> None:
    """漏配不是致命错误（退回默认档），但必须能被发现——否则静默降级到贵模型。"""
    router = _router({ROUTE_ANALYST_SQL: TIER_FAST})
    assert router.tier_for(ROUTE_PLANNER) == TIER_REASONING
    assert router.unmapped_routes == {ROUTE_PLANNER}


def test_none_route_uses_default_tier_without_being_flagged() -> None:
    router = _router()
    assert router.tier_for(None) == TIER_REASONING
    assert router.unmapped_routes == set()


def test_route_pointing_to_missing_tier_raises() -> None:
    """配置写错档位名必须立刻报错，不能悄悄用默认档。"""
    router = _router({ROUTE_PLANNER: "typo-tier"})
    with pytest.raises(KeyError, match="typo-tier"):
        router.tier_for(ROUTE_PLANNER)


def test_router_requires_at_least_one_tier() -> None:
    with pytest.raises(ValueError):
        ModelRouter(tiers={})


def test_default_tier_falls_back_to_first_when_config_is_wrong() -> None:
    router = ModelRouter(tiers={TIER_FAST: _gateway("fast")}, default_tier="nonexistent")
    assert router.default_tier == TIER_FAST


# ------------------------------------------------------------------ 实际分派


def test_complete_dispatches_to_configured_tier() -> None:
    router = _router({ROUTE_ANALYST_SQL: TIER_FAST})
    response = asyncio.run(
        router.complete([LLMMessage.user("hi")], route=ROUTE_ANALYST_SQL)
    )
    assert response.provider == "fast", response.provider


def test_complete_without_route_uses_default_tier() -> None:
    router = _router({ROUTE_ANALYST_SQL: TIER_FAST})
    response = asyncio.run(router.complete([LLMMessage.user("hi")]))
    assert response.provider == "reasoning"


def test_two_call_sites_can_land_on_different_tiers_in_one_task() -> None:
    """同一个任务里，取数走 fast、归因走 reasoning —— 这正是分级路由的目的。"""
    router = _router({ROUTE_ANALYST_SQL: TIER_FAST})
    first = asyncio.run(router.complete([LLMMessage.user("sql")], route=ROUTE_ANALYST_SQL))
    second = asyncio.run(router.complete([LLMMessage.user("conclude")], route="analyst_conclude"))
    assert first.provider == "fast"
    assert second.provider == "reasoning"


# ------------------------------------------------------------------ 归因明细


def test_history_is_chronological_and_carries_route() -> None:
    router = _router({ROUTE_ANALYST_SQL: TIER_FAST})
    asyncio.run(router.complete([LLMMessage.user("a")], route=ROUTE_PLANNER))
    asyncio.run(router.complete([LLMMessage.user("b")], route=ROUTE_ANALYST_SQL))
    asyncio.run(router.complete([LLMMessage.user("c")], route=ROUTE_PLANNER))

    assert [r.route for r in router.history] == [
        ROUTE_PLANNER,
        ROUTE_ANALYST_SQL,
        ROUTE_PLANNER,
    ], "明细必须按发生顺序，否则无法归因时序"


def test_history_records_tokens_and_duration() -> None:
    router = _router()
    asyncio.run(router.complete([LLMMessage.user("hi")], route=ROUTE_PLANNER))
    record = router.history[-1]
    assert record.total_tokens > 0
    assert record.duration_ms >= 0
    assert record.input_tokens > 0 and record.output_tokens > 0


def test_summarize_by_route_aggregates_per_call_site() -> None:
    router = _router({ROUTE_ANALYST_SQL: TIER_FAST})
    asyncio.run(router.complete([LLMMessage.user("a")], route=ROUTE_ANALYST_SQL))
    asyncio.run(router.complete([LLMMessage.user("b")], route=ROUTE_ANALYST_SQL))
    asyncio.run(router.complete([LLMMessage.user("c")], route=ROUTE_PLANNER))

    summary = router.summarize_by_route()
    assert summary[ROUTE_ANALYST_SQL]["calls"] == 2
    assert summary[ROUTE_PLANNER]["calls"] == 1
    assert summary[ROUTE_ANALYST_SQL]["input_tokens"] > 0
    assert summary[ROUTE_ANALYST_SQL]["total_tokens"] > summary[ROUTE_PLANNER]["total_tokens"]


def test_failed_call_is_also_recorded_with_route() -> None:
    from multiagent.gateway import AllProvidersFailedError

    router = ModelRouter(
        tiers={TIER_REASONING: LLMGateway(primary=StubProvider(name="p", should_fail=True))}
    )
    with pytest.raises(AllProvidersFailedError):
        asyncio.run(router.complete([LLMMessage.user("hi")], route=ROUTE_PLANNER))
    assert router.history[-1].ok is False
    assert router.history[-1].route == ROUTE_PLANNER


# ------------------------------------------------------- 与 LLMGateway 可互换


def test_router_and_gateway_are_signature_compatible() -> None:
    """业务侧只依赖 complete() 签名，两者必须可互换。"""
    import inspect

    gateway_params = set(inspect.signature(LLMGateway.complete).parameters)
    router_params = set(inspect.signature(ModelRouter.complete).parameters)
    assert gateway_params == router_params, (gateway_params, router_params)


def test_build_gateway_from_providers_honours_stub_mode() -> None:
    gateway = build_gateway_from_providers(
        "stub", settings.fast_primary, settings.fast_fallback
    )
    response = asyncio.run(gateway.complete([LLMMessage.user("hi")]))
    assert response.provider.startswith("stub:"), response.provider


# --------------------------------------------------------------- 配置装配


def test_build_router_from_settings_creates_two_tiers() -> None:
    router = build_router_from_settings(Settings(_env_file=None))
    assert set(router.tiers) == {TIER_REASONING, TIER_FAST}


def test_default_settings_route_everything_to_reasoning() -> None:
    """引入路由机制的阶段一必须零行为变化：默认全部走 reasoning 档。"""
    router = build_router_from_settings(Settings(_env_file=None))
    for route in (ROUTE_PLANNER, ROUTE_ANALYST_SQL, "analyst_conclude"):
        assert router.tier_for(route) == TIER_REASONING, route
    assert router.unmapped_routes == set()


def test_routing_can_be_overridden_from_settings() -> None:
    settings_override = Settings(
        _env_file=None, llm_routing={ROUTE_ANALYST_SQL: TIER_FAST}
    )
    router = build_router_from_settings(settings_override)
    assert router.tier_for(ROUTE_ANALYST_SQL) == TIER_FAST
    response = asyncio.run(
        router.complete([LLMMessage.user("hi")], route=ROUTE_ANALYST_SQL)
    )
    assert response.provider.startswith("stub:zhipu"), response.provider


def test_settings_tier_for_matches_router_logic_on_same_config() -> None:
    """给定**同一份路由配置**，Settings 与 ModelRouter 的判定必须一致——
    两者不一致会让人照文档配置却得到别的行为。"""
    override = Settings(_env_file=None, llm_routing={ROUTE_ANALYST_SQL: TIER_FAST})
    router = build_router_from_settings(override)

    assert override.tier_for(ROUTE_ANALYST_SQL) == TIER_FAST
    assert router.tier_for(ROUTE_ANALYST_SQL) == TIER_FAST
    assert override.tier_for("没配过的调用点") == router.tier_for("没配过的调用点")
    assert router.tier_for("没配过的调用点") == override.llm_default_tier


def test_default_settings_fast_tier_points_at_local_non_reasoning_endpoint() -> None:
    """fast 档默认应指向低延迟端点，否则分级路由没有意义。"""
    defaults = Settings(_env_file=None)
    assert defaults.fast_primary.base_url == "http://127.0.0.1:7863/v1"
    assert defaults.fast_fallback.provider == "deepseek"


# ------------------------------------------------- 输出上限钳制（fast 档延时的主控制点）


class _RecordingProvider(StubProvider):
    """记录每次调用实际拿到的 max_tokens。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seen_max_tokens: list[int | None] = []

    async def complete(self, messages, *, max_tokens=None, temperature=None):
        self.seen_max_tokens.append(max_tokens)
        return await super().complete(messages, max_tokens=max_tokens, temperature=temperature)


def _cap_router(caps: dict[str, int]):
    recorder = _RecordingProvider(name="rec")
    router = ModelRouter(
        tiers={TIER_FAST: LLMGateway(primary=recorder)},
        default_tier=TIER_FAST,
        output_caps=caps,
    )
    return router, recorder


def test_router_clamps_oversized_max_tokens_down_to_tier_cap() -> None:
    """给本地模型过大的输出上限等于邀请它无限思考——必须在网关层掐掉。"""
    router, recorder = _cap_router({TIER_FAST: 2048})
    asyncio.run(router.complete([LLMMessage.user("hi")], max_tokens=8192))
    assert recorder.seen_max_tokens == [2048]


def test_router_keeps_smaller_request_untouched() -> None:
    """调用方要得更少时不应被放大。"""
    router, recorder = _cap_router({TIER_FAST: 2048})
    asyncio.run(router.complete([LLMMessage.user("hi")], max_tokens=512))
    assert recorder.seen_max_tokens == [512]


def test_router_applies_tier_cap_when_caller_passes_none() -> None:
    router, recorder = _cap_router({TIER_FAST: 2048})
    asyncio.run(router.complete([LLMMessage.user("hi")]))
    assert recorder.seen_max_tokens == [2048]


def test_router_without_cap_for_tier_does_not_clamp() -> None:
    router, recorder = _cap_router({})
    asyncio.run(router.complete([LLMMessage.user("hi")], max_tokens=8192))
    assert recorder.seen_max_tokens == [8192]


def test_cap_follows_route_mapping() -> None:
    """钳制必须随路由走：切到 fast 的调用点才被收紧。"""
    fast_rec, reason_rec = _RecordingProvider(name="fast"), _RecordingProvider(name="reason")
    router = ModelRouter(
        tiers={TIER_FAST: LLMGateway(primary=fast_rec), TIER_REASONING: LLMGateway(primary=reason_rec)},
        routing={ROUTE_ANALYST_SQL: TIER_FAST, ROUTE_PLANNER: TIER_REASONING},
        output_caps={TIER_FAST: 2048, TIER_REASONING: 8192},
    )
    asyncio.run(router.complete([LLMMessage.user("a")], max_tokens=99999, route=ROUTE_ANALYST_SQL))
    asyncio.run(router.complete([LLMMessage.user("b")], max_tokens=99999, route=ROUTE_PLANNER))
    assert fast_rec.seen_max_tokens == [2048]
    assert reason_rec.seen_max_tokens == [8192]


def test_build_router_from_settings_wires_caps_from_settings() -> None:
    override = Settings(_env_file=None, llm_fast_max_output_tokens=1024, llm_max_output_tokens=4096)
    router = build_router_from_settings(override)
    assert router.output_caps[TIER_FAST] == 1024
    assert router.output_caps[TIER_REASONING] == 4096
    # 默认路由表下 analyst_sql 走 reasoning，应拿到 reasoning 的上限
    assert router.output_cap_for(ROUTE_ANALYST_SQL) == 4096


def test_fast_tier_cap_is_smaller_than_reasoning_cap_by_default() -> None:
    defaults = Settings(_env_file=None)
    assert defaults.llm_fast_max_output_tokens < defaults.llm_max_output_tokens


# ---------------------------------------- 截断时放宽一次（收紧上限的配套安全网）


class _TruncatingProvider(LLMProvider):
    """复现真实失效模式：额度不足时把额度全烧在「思考」上，返回空内容。

    本地模型实测就是这样——给 2048 上限时它思考 2048 token 后正文为空。
    """

    def __init__(self, name: str = "trunc", enough: int = 4096) -> None:
        self.name = name
        self.model = "trunc-model"
        self.enough = enough
        self.seen_max_tokens: list[int | None] = []

    async def complete(self, messages, *, max_tokens=None, temperature=None) -> LLMResponse:
        self.seen_max_tokens.append(max_tokens)
        if max_tokens is None or max_tokens < self.enough:
            raise EmptyCompletionError(self.name, "length", max_tokens or 0)
        return LLMResponse(
            content="ok", provider=self.name, model=self.model, input_tokens=1, output_tokens=1
        )

    async def health(self) -> bool:
        return True


def _truncating_router(enough: int = 4096):
    provider = _TruncatingProvider(enough=enough)
    router = ModelRouter(
        tiers={TIER_FAST: LLMGateway(primary=provider)},
        default_tier=TIER_FAST,
        output_caps={TIER_FAST: 2048},
    )
    return router, provider


def test_truncated_fast_call_escalates_once_and_succeeds() -> None:
    """因上限被思考吃满而失败，放宽一次即恢复——不应让任务失败。"""
    router, provider = _truncating_router()
    response = asyncio.run(router.complete([LLMMessage.user("hi")], max_tokens=8192))
    assert response.content == "ok"
    assert provider.seen_max_tokens == [2048, 8192], provider.seen_max_tokens


def test_escalation_is_not_attempted_when_caller_budget_is_already_the_cap() -> None:
    """调用方原本就只要 1024，则没有「更宽」的额度可放，应如实抛错而不是偷偷放宽。"""
    router, provider = _truncating_router()
    with pytest.raises(AllProvidersFailedError):
        asyncio.run(router.complete([LLMMessage.user("hi")], max_tokens=1024))
    assert provider.seen_max_tokens == [1024], provider.seen_max_tokens


def test_non_truncation_failure_is_not_escalated() -> None:
    """供应商真挂了就不该靠放宽额度掩盖——必须如实抛出。"""
    router = ModelRouter(
        tiers={TIER_REASONING: LLMGateway(primary=StubProvider(name="p", should_fail=True))},
        default_tier=TIER_REASONING,
        output_caps={TIER_REASONING: 2048},
    )
    with pytest.raises(AllProvidersFailedError):
        asyncio.run(router.complete([LLMMessage.user("hi")], max_tokens=8192))
    assert len(router.history) == 1, "非截断失败不应触发额外调用"


def test_escalation_records_both_attempts_in_history() -> None:
    router, _ = _truncating_router()
    asyncio.run(router.complete([LLMMessage.user("hi")], max_tokens=8192))
    assert len(router.history) == 2
    assert router.history[0].ok is False
    assert router.history[1].ok is True
    assert all(r.route is None for r in router.history)


def test_all_truncated_detection_uses_exception_types_not_text() -> None:
    from multiagent.gateway.router import _all_truncated

    truncated = AllProvidersFailedError(
        [("a", "x")], [EmptyCompletionError("a", "length", 2048)]
    )
    unrelated = AllProvidersFailedError(
        [("a", "y")], [ProviderUnavailableError("a", "network down")]
    )
    mixed = AllProvidersFailedError(
        [("a", "x"), ("b", "y")],
        [EmptyCompletionError("a", "length", 2048), ProviderUnavailableError("b", "down")],
    )
    assert _all_truncated(truncated) is True
    assert _all_truncated(unrelated) is False
    assert _all_truncated(mixed) is False
