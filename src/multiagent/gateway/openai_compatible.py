"""OpenAI 兼容供应商适配器。

DeepSeek、通义千问（DashScope 兼容模式）、智谱 GLM 等都提供 OpenAI 兼容的
`POST {base_url}/chat/completions`，因此共用本实现。换供应商只需改配置。

安全约定：API key 只从配置注入，**任何情况下不写入日志、不进入异常消息**。
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import (
    EmptyCompletionError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ProviderUnavailableError,
)

# 单次调用（非流式）的读超时。
# W6 实测：本地推理模型在重负载下单次调用可超过 60 秒——60 秒会把「慢但活着」的
# 调用掐死，而 reasoning 档的主备指向同一端点，于是两次都超时、整个任务失败
# （评测里 27 条中 1 条因此失败）。120 秒给慢调用留出完成空间；死端点的代价是
# 等更久，属可接受（W7 的异常处理体系会补上重试与超时策略）。
DEFAULT_TIMEOUT_SECONDS = 120.0


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = provider
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = (api_key or "").strip()
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    # ---- 内部工具 ----
    def _url(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        # 只在请求头里带 key，绝不回显
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _client_or_new(self) -> tuple[httpx.AsyncClient, bool]:
        if self._client is not None:
            return self._client, False
        return httpx.AsyncClient(timeout=self._timeout), True

    # ---- LLMProvider ----
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not self._api_key:
            raise ProviderUnavailableError(
                self.name, "未配置 API key（请在 .env 中设置或使用 LLM_MODE=stub）"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        client, owns = await self._client_or_new()
        try:
            resp = await client.post(self._url(), headers=self._headers(), json=payload)
            if resp.status_code >= 400:
                raise ProviderUnavailableError(
                    self.name, f"HTTP {resp.status_code}: {self._safe_body(resp)}"
                )
            data = resp.json()
        except ProviderUnavailableError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(self.name, f"请求超时: {exc}") from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(self.name, f"网络错误: {exc}") from exc
        except ValueError as exc:  # 响应不是合法 JSON
            raise ProviderUnavailableError(self.name, f"响应解析失败: {exc}") from exc
        finally:
            if owns:
                await client.aclose()

        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailableError(self.name, f"响应结构不符合预期: {exc}") from exc

        usage = data.get("usage") or {}
        # 推理型模型把思考 token 单列在 completion_tokens_details 里，
        # 但它已计入 completion_tokens，故 total 不会重复计算。
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
        finish_reason = choice.get("finish_reason")

        if not content.strip():
            # 空内容绝不静默返回——推理型模型在额度不足时会走到这里。
            raise EmptyCompletionError(self.name, finish_reason, reasoning_tokens)

        return LLMResponse(
            content=content,
            provider=self.name,
            model=data.get("model") or self.model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=finish_reason,
            reasoning_tokens=reasoning_tokens,
        )

    @staticmethod
    def _safe_body(resp: httpx.Response, limit: int = 300) -> str:
        """截断错误响应体；供应商报错体里偶尔含 key 片段，故不整体回显。"""
        text = resp.text or ""
        return text[:limit]

    async def health(self) -> bool:
        if not self._api_key:
            return False
        try:
            await self.complete([LLMMessage.user("ping")], max_tokens=1)
        except ProviderUnavailableError:
            return False
        return True
