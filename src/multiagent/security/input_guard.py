"""输入侧安全过滤（W9）：Prompt Injection 规则 + 敏感词 + PII 就地脱敏。

语义：

- **命中注入规则 / 敏感词 → 拒绝**（`allowed=False`），调用方返回稳定错误码
  `INPUT_BLOCKED`，不创建任务、不调模型。
- **文本里的身份证/手机号/银行卡/密钥 → 就地脱敏后放行**：用户可能只是把
  数据贴进来做分析，直接拒绝会让正常需求无法完成；脱敏保证模型看不到原值。
- 规则做 **NFKC + 去零宽字符 + 空白折叠 + 小写** 归一化，大小写、全角、
  零宽字符和多余空格不能绕过。

规则宁精勿滥：只拦「明确的越权指令」和「索要凭据」的表述，不拦正常的
「分析系统提示词对效果的影响」这类业务问题。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .dlp import redact_text

#: 默认敏感词：只收「凭据 / 身份标识」类词，避免误伤正常分析问题。
#: 可通过 `SECURITY_BLOCKED_TERMS`（逗号分隔）覆盖。
DEFAULT_BLOCKED_TERMS: tuple[str, ...] = (
    "密码",
    "口令",
    "私钥",
    "api key",
    "apikey",
    "secret key",
    "access token",
    "身份证号",
    "银行卡号",
    "信用卡号",
    "cvv",
)


@dataclass(frozen=True)
class InputVerdict:
    allowed: bool
    text: str
    blocked_reason: str | None = None
    rule: str | None = None
    redactions: int = 0
    matched: tuple[str, ...] = field(default_factory=tuple)


#: (规则 id, 正则)。英文按词边界，中文按短语；都在归一化文本上匹配。
_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous",
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?"
            r"(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)"
        ),
    ),
    (
        "ignore_previous_zh",
        re.compile(
            r"(?:忽略|无视|忘记|不要理会|别管).{0,12}"
            r"(?:之前|以上|前面|上述|先前).{0,8}(?:指令|提示|要求|规则)"
        ),
    ),
    (
        "reveal_prompt",
        re.compile(
            r"(?:输出|显示|告诉我|打印|泄露|复述|reveal|show|print|leak).{0,16}"
            r"(?:系统提示|提示词|system\s+prompt|hidden\s+prompt|instructions?)"
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"(?:你现在是|从现在开始你是|进入|开启|you\s+are\s+now|act\s+as|enter)"
            r".{0,12}(?:无限制|不受限|开发者模式|越狱|jailbreak|\bdan\b|developer\s+mode)"
        ),
    ),
    (
        "bypass_safety",
        re.compile(
            r"(?:忽略|绕过|关闭|解除|disable|bypass|turn\s+off).{0,12}"
            r"(?:安全|限制|沙箱|过滤|审查|safety|filter|sandbox|guardrail)"
        ),
    ),
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for char in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(char, "")
    # 中文短语里插空格是常见绕过手法（「忽 略 指 令」）；只去掉
    # **两个汉字之间**的空白，保留英文词边界。
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def scan_input(
    text: str,
    *,
    blocked_terms: tuple[str, ...] | list[str] | None = None,
    enabled: bool = True,
) -> InputVerdict:
    """扫描用户输入；返回是否放行、脱敏后的文本与命中信息。"""
    if not enabled:
        return InputVerdict(allowed=True, text=text)

    normalized = _normalize(text)
    for rule_id, pattern in _INJECTION_RULES:
        if pattern.search(normalized):
            return InputVerdict(
                allowed=False,
                text="",
                blocked_reason=f"检测到疑似 Prompt Injection（规则 {rule_id}）",
                rule=rule_id,
            )

    terms = tuple(blocked_terms) if blocked_terms is not None else DEFAULT_BLOCKED_TERMS
    for term in terms:
        needle = _normalize(term)
        if needle and needle in normalized:
            return InputVerdict(
                allowed=False,
                text="",
                blocked_reason=f"输入包含敏感词「{term}」，为安全起见已拒绝",
                rule="sensitive_term",
                matched=(term,),
            )

    redacted, count = redact_text(text)
    return InputVerdict(allowed=True, text=redacted, redactions=count)
