"""LLM 网关测试：主备降级、错误处理、密钥不外泄。

live 路径用 httpx.MockTransport 模拟，因此**不需要 API key、不需要网络**
也能验证真实的请求构造与错误收敛。

运行方式（本机未装 pytest 时）：
    python tests/test_gateway.py
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from multiagent.config import Settings  # noqa: E402
from multiagent.gateway import (  # noqa: E402
    AllProvidersFailedError,
    EmptyCompletionError,
    LLMGateway,
    LLMMessage,
    OpenAICompatibleProvider,
    ProviderUnavailableError,
    StubProvider,
    build_gateway,
)
from multiagent.gateway.gateway import build_gateway_from_settings  # noqa: E402

FAKE_KEY = "sk-SECRET-DO-NOT-LEAK-1234567890"

_OK_BODY = {
    "model": "deepseek-chat",
    "choices": [{"message": {"role": "assistant", "content": "分析结论"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 45},
}


def _run_async(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ Stub / 降级


def test_stub_gateway_returns_response() -> None:
    gw = build_gateway(
        mode="stub",
        primary_provider="deepseek", primary_model="deepseek-chat", primary_base_url="x",
        primary_api_key=None,
        fallback_provider="qwen", fallback_model="qwen-plus", fallback_base_url="y",
        fallback_api_key=None,
    )
    resp = _run_async(gw.complete([LLMMessage.user("你好")]))
    assert resp.provider == "stub:deepseek", resp.provider
    assert resp.total_tokens > 0
    assert gw.history[-1].degraded is False


def test_degrades_to_fallback_when_primary_fails() -> None:
    gw = LLMGateway(
        primary=StubProvider(name="primary", should_fail=True),
        fallback=StubProvider(name="fallback"),
    )
    resp = _run_async(gw.complete([LLMMessage.user("hi")]))
    assert resp.provider == "fallback"
    record = gw.history[-1]
    assert record.degraded is True
    assert record.chain == ["primary", "fallback"], record.chain


def test_all_providers_failed_collects_every_reason() -> None:
    gw = LLMGateway(
        primary=StubProvider(name="p1", should_fail=True, fail_reason="原因一"),
        fallback=StubProvider(name="p2", should_fail=True, fail_reason="原因二"),
    )
    try:
        _run_async(gw.complete([LLMMessage.user("hi")]))
    except AllProvidersFailedError as exc:
        reasons = dict(exc.failures)
        assert reasons["p1"] == "原因一" and reasons["p2"] == "原因二", reasons
        assert gw.history[-1].ok is False
        return
    raise AssertionError("主备全失败时应抛 AllProvidersFailedError")


# ------------------------------------------------- openai 兼容适配器（MockTransport）


def _provider_with_handler(handler, *, api_key: str | None = FAKE_KEY) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider(
        provider="deepseek", model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=api_key, client=client,
    )


def test_openai_compatible_sends_correct_request_and_parses() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_OK_BODY)

    resp = _run_async(_provider_with_handler(handler).complete([LLMMessage.user("你好")]))

    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions", captured["url"]
    assert captured["auth"] == f"Bearer {FAKE_KEY}"
    assert captured["body"]["model"] == "deepseek-chat"
    assert captured["body"]["messages"] == [{"role": "user", "content": "你好"}]
    assert resp.content == "分析结论"
    assert (resp.input_tokens, resp.output_tokens) == (120, 45)


def test_missing_api_key_raises_without_calling_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("没有 key 时不应发起请求")

    provider = _provider_with_handler(handler, api_key="")
    try:
        _run_async(provider.complete([LLMMessage.user("hi")]))
    except ProviderUnavailableError as exc:
        assert "API key" in str(exc)
        return
    raise AssertionError("未配置 key 时应抛 ProviderUnavailableError")


def test_http_error_message_does_not_leak_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    provider = _provider_with_handler(handler)
    try:
        _run_async(provider.complete([LLMMessage.user("hi")]))
    except ProviderUnavailableError as exc:
        message = str(exc)
        assert "401" in message, message
        assert FAKE_KEY not in message, "异常信息不得包含 API key"
        return
    raise AssertionError("HTTP 401 时应抛 ProviderUnavailableError")


def test_network_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider_with_handler(handler)
    try:
        _run_async(provider.complete([LLMMessage.user("hi")]))
    except ProviderUnavailableError as exc:
        assert "网络错误" in str(exc), str(exc)
        return
    raise AssertionError("网络异常应被包装为 ProviderUnavailableError")


def test_live_mode_without_keys_degrades_then_fails_cleanly() -> None:
    gw = build_gateway(
        mode="live",
        primary_provider="deepseek", primary_model="deepseek-chat", primary_base_url="https://api.deepseek.com/v1",
        primary_api_key=None,
        fallback_provider="qwen", fallback_model="qwen-plus", fallback_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        fallback_api_key=None,
    )
    try:
        _run_async(gw.complete([LLMMessage.user("hi")]))
    except AllProvidersFailedError as exc:
        assert len(exc.failures) == 2, exc.failures
        assert all("API key" in reason for _, reason in exc.failures)
        return
    raise AssertionError("live 模式且无 key 时应在两次降级后抛错")


# ------------------------------------------------- 推理型模型的额度耗尽陷阱
#
# deepseek-flash 等推理型模型先产出 reasoning_content 再产出 content，
# max_tokens 由两者共享。额度偏小时 content 为空且 finish_reason=length。
# 这类结果必须显式报错，不得静默返回空字符串。

_REASONING_ONLY_BODY = {
    "model": "deepseek-flash",
    "choices": [
        {
            "message": {"role": "assistant", "content": "", "reasoning_content": "The user wants..."},
            "finish_reason": "length",
        }
    ],
    "usage": {
        "prompt_tokens": 35,
        "completion_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": 20},
    },
}

_REASONING_WITH_CONTENT_BODY = {
    "model": "deepseek-flash",
    "choices": [
        {
            "message": {"role": "assistant", "content": "正常", "reasoning_content": "The user wants..."},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 35,
        "completion_tokens": 28,
        "completion_tokens_details": {"reasoning_tokens": 26},
    },
}

_REASONING_TRUNCATED_WITH_CONTENT_BODY = {
    "model": "deepseek-flash",
    "choices": [
        {
            "message": {"role": "assistant", "content": "部分结论，未写完", "reasoning_content": "..."},
            "finish_reason": "length",
        }
    ],
    "usage": {
        "prompt_tokens": 35,
        "completion_tokens": 60,
        "completion_tokens_details": {"reasoning_tokens": 20},
    },
}


def test_budget_exhausted_by_reasoning_raises_instead_of_silent_empty() -> None:
    provider = _provider_with_handler(lambda req: httpx.Response(200, json=_REASONING_ONLY_BODY))
    try:
        _run_async(provider.complete([LLMMessage.user("只回复两个字：正常")], max_tokens=20))
    except EmptyCompletionError as exc:
        assert exc.reasoning_tokens == 20, exc.reasoning_tokens
        assert exc.finish_reason == "length"
        assert "max_tokens" in str(exc), "错误信息应提示调大 max_tokens"
        return
    raise AssertionError("额度被思考耗尽导致 content 为空时必须报错，不得静默返回空串")


def test_reasoning_tokens_recorded_for_cost_metering() -> None:
    provider = _provider_with_handler(
        lambda req: httpx.Response(200, json=_REASONING_WITH_CONTENT_BODY)
    )
    resp = _run_async(provider.complete([LLMMessage.user("hi")]))
    assert resp.content == "正常"
    assert resp.reasoning_tokens == 26
    assert resp.output_tokens == 28, "reasoning_tokens 已计入 completion_tokens，不应重复累加"
    assert resp.total_tokens == 63
    assert resp.truncated is False


def test_truncated_flag_surfaces_partial_output() -> None:
    provider = _provider_with_handler(
        lambda req: httpx.Response(200, json=_REASONING_TRUNCATED_WITH_CONTENT_BODY)
    )
    resp = _run_async(provider.complete([LLMMessage.user("hi")], max_tokens=60))
    assert resp.truncated is True, "finish_reason=length 时应标记为截断"
    assert resp.content


def test_empty_completion_triggers_fallback_to_other_provider() -> None:
    gw = LLMGateway(
        primary=_provider_with_handler(
            lambda req: httpx.Response(200, json=_REASONING_ONLY_BODY)
        ),
        fallback=StubProvider(name="fallback"),
    )
    resp = _run_async(gw.complete([LLMMessage.user("hi")], max_tokens=20))
    assert resp.provider == "fallback"
    assert gw.history[-1].degraded is True
    assert gw.history[-1].chain == ["deepseek", "fallback"]


# ---------------------------------------------------------------------- 配置


def test_settings_isolated_from_env_file_default_to_stub() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_mode == "stub"
    assert settings.primary.has_key is False
    assert settings.primary.provider == "deepseek"


def test_build_gateway_from_settings_uses_stub_mode() -> None:
    gw = build_gateway_from_settings(Settings(_env_file=None))
    resp = _run_async(gw.complete([LLMMessage.user("hi")]))
    assert resp.provider.startswith("stub:"), resp.provider


# --------------------------------------------------------------------- 运行器


def _run() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
