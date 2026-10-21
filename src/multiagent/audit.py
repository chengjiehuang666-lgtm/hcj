"""审计日志（W14）：只追加 JSONL，记录谁在什么时候做了什么。

与普通结构化日志分开：
- 普通日志是**观测**（模型调用、耗时），可采样、可丢弃；
- 审计日志是**责任链**（认证失败、提交、审批、任务结束、限流），只追加、不修改。

只记录安全相关字段：actor（Key 指纹/角色）、task_id、事件、结果；
**不记录原始问题、Key、模型输入输出**——需要定位时用 `question_sha256` 短哈希。
写失败不阻塞业务，但会在普通日志里报 `audit_write_failed`。
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .observability import log_event
from .security.dlp import redact_structure


class AuditLogger:
    def __init__(self, settings: Any) -> None:
        self.enabled = bool(getattr(settings, "audit_enabled", False))
        self.path = Path(getattr(settings, "audit_log_path", "audit/audit.jsonl"))
        self._lock = threading.Lock()

    @staticmethod
    def question_digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def write(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "ts": datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
            "event": event,
            **{k: v for k, v in redact_structure(fields).items() if v is not None},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False, default=str)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
        except Exception as exc:  # noqa: BLE001 — 审计写失败不能拖垮任务
            log_event(
                "audit_write_failed",
                level=30,
                error=f"{type(exc).__name__}: {exc}"[:200],
                audit_path=str(self.path),
            )
