"""输出侧 DLP：正则脱敏（W9）。

覆盖《实施路线图》W9 明确列出的四类：**身份证、手机号、银行卡、密钥**。

设计取舍：

- **只脱敏、不阻断**：输出已经产生，阻断会让任务白跑；脱敏保留其余信息。
- **银行卡用 Luhn 校验**：仅靠 16–19 位数字会误伤营收/订单号（演示数据里
  2335000 这类数字很常见）；Luhn 通过才认为是卡号。
- **幂等**：脱敏标记本身不含数字，重复调用不会叠加替换。
- 脱敏发生在**用户可见边界**（API 结果 / SSE / CLI / 记忆沉淀），
  内部执行状态保留原值——下游节点需要真实数据才能继续推理。
"""

from __future__ import annotations

import re
from typing import Any

# 顺序很重要：私钥块 → 各类令牌 → 身份证 → 手机号 → 银行卡。
# 先处理长/高置信模式，避免被短模式截断。
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"), "[密钥已脱敏]"),
    (re.compile(r"\brk-[A-Za-z0-9_\-]{12,}\b"), "[密钥已脱敏]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[令牌已脱敏]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[令牌已脱敏]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[密钥已脱敏]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[令牌已脱敏]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[令牌已脱敏]"),
    (
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password|passwd)"
            r"\b\s*[:=]\s*[^\s,;]+"
        ),
        "[凭据已脱敏]",
    ),
]
_ID_CARD = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)
_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
# 16–19 位、首位 2–6、允许单空格/连字符分组；是否真是卡号交给 Luhn。
_CARD_CANDIDATE = re.compile(r"(?<!\d)[2-6](?:[ -]?\d){15,18}(?!\d)")

_ID_MARK = "[身份证已脱敏]"
_PHONE_MARK = "[手机号已脱敏]"
_CARD_MARK = "[银行卡已脱敏]"


def _luhn_ok(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = ord(char) - 48
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _redact_cards(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        digits = re.sub(r"[ -]", "", match.group(0))
        if _luhn_ok(digits):
            count += 1
            return _CARD_MARK
        return match.group(0)

    return _CARD_CANDIDATE.sub(replace, text), count


def redact_text(text: str) -> tuple[str, int]:
    """对一段文本做 DLP 脱敏；返回 (脱敏后文本, 替换次数)。"""
    if not isinstance(text, str) or not text:
        return text, 0
    result, total = _PRIVATE_KEY.subn("[私钥已脱敏]", text)
    for pattern, marker in _SECRET_PATTERNS:
        result, hits = pattern.subn(marker, result)
        total += hits
    result, hits = _ID_CARD.subn(_ID_MARK, result)
    total += hits
    result, hits = _PHONE.subn(_PHONE_MARK, result)
    total += hits
    result, hits = _redact_cards(result)
    total += hits
    return result, total


def redact_structure(value: Any) -> Any:
    """递归脱敏 dict / list / tuple 中的字符串值（键保持原样）。"""
    if isinstance(value, str):
        return redact_text(value)[0]
    if isinstance(value, dict):
        return {key: redact_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_structure(item) for item in value)
    return value


def redact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """记忆条目落库前的脱敏（内容 + 来源引用）。"""
    redacted = dict(entry)
    if isinstance(redacted.get("content"), str):
        redacted["content"] = redact_text(redacted["content"])[0]
    if isinstance(redacted.get("source_ref"), str):
        redacted["source_ref"] = redact_text(redacted["source_ref"])[0]
    return redacted
