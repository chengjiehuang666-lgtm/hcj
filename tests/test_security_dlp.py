"""W9 输出侧 DLP：身份证 / 手机号 / 银行卡 / 密钥脱敏。"""

from __future__ import annotations

from multiagent.security.dlp import redact_entry, redact_structure, redact_text


def test_redacts_id_phone_card_and_secret() -> None:
    text = (
        "身份证 11010519491231002X，手机 13812345678，"
        "银行卡 4111111111111111，密钥 sk-abcdefghijklmnopqrstuvwx"
    )
    redacted, count = redact_text(text)
    assert "11010519491231002X" not in redacted
    assert "13812345678" not in redacted
    assert "4111111111111111" not in redacted
    assert "sk-abcdefghijklmnopqrstuvwx" not in redacted
    assert "[身份证已脱敏]" in redacted
    assert "[手机号已脱敏]" in redacted
    assert "[银行卡已脱敏]" in redacted
    assert "[密钥已脱敏]" in redacted
    assert count >= 4


def test_does_not_redact_normal_business_numbers() -> None:
    text = "华东区 9 月销售额 2335000，环比 -20.85%，订单 11400 单，日期 2026-09-30"
    redacted, count = redact_text(text)
    assert redacted == text
    assert count == 0


def test_bank_card_requires_luhn() -> None:
    # 16 位但 Luhn 不通过：不应被当作银行卡
    invalid = "1234567890123456"
    redacted, count = redact_text(invalid)
    assert redacted == invalid and count == 0
    # 合法 Visa 测试卡
    redacted, count = redact_text("4111 1111 1111 1111")
    assert redacted == "[银行卡已脱敏]" and count == 1


def test_secret_and_private_key_patterns() -> None:
    text = (
        "AKIAABCDEFGHIJKLMNOP ghp_abcdefghijklmnopqrstuvwxyz0123456789 "
        "password: hunter2\n"
        "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
    )
    redacted, _ = redact_text(text)
    for leaked in ("AKIAABCDEFGHIJKLMNOP", "ghp_", "hunter2", "MIIE"):
        assert leaked not in redacted
    assert "[密钥已脱敏]" in redacted or "[令牌已脱敏]" in redacted


def test_redaction_is_idempotent() -> None:
    once, _ = redact_text("手机 13812345678")
    twice, count = redact_text(once)
    assert twice == once
    assert count == 0


def test_redact_structure_recurses() -> None:
    payload = {
        "summary": "联系 13812345678",
        "items": ["身份证 11010519491231002X", {"secret": "sk-abcdefghijklmnopqrstuvwx"}],
        "rows": 3,
    }
    redacted = redact_structure(payload)
    assert "13812345678" not in redacted["summary"]
    assert "11010519491231002X" not in redacted["items"][0]
    assert "sk-abcdefghijklmnopqrstuvwx" not in redacted["items"][1]["secret"]
    assert redacted["rows"] == 3


def test_redact_entry_cleans_content_and_source() -> None:
    entry = {
        "task_id": "t",
        "content": "用户 13812345678 的结论",
        "source_ref": "任务 t，手机 13812345678",
    }
    cleaned = redact_entry(entry)
    assert "13812345678" not in cleaned["content"]
    assert "13812345678" not in cleaned["source_ref"]
    assert cleaned["task_id"] == "t"
