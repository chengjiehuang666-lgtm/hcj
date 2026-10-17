"""任务注册表：提交、进度事件、结果查询。

**当前是进程内存储**，进程重启即丢失。这与「任务状态外置、可断点续跑」的目标形态
（见《系统架构设计》2.2.1）尚有距离——MVP 阶段先用最简实现，把接口形态固定下来；
改为 Redis/PostgreSQL 持久化属后续版本（W7 异常处理一并做）。

**为什么仍要保留事件缓冲**：SSE 客户端可能在任务开始后才连上来，
若只推增量就会漏掉前面的进度；缓冲 + 补发让晚连接的客户端也能看到完整过程。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# 终态：不再产生新事件
# 终态：不再产生新事件。awaiting_approval 是**非终态**（HITL 暂停）
TERMINAL_STATUSES = frozenset({"ok", "failed", "rejected"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class TaskRecord:
    task_id: str
    question: str
    # pending | running | awaiting_approval | ok | failed | rejected
    status: str = "pending"
    created_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    #: 原始 TaskEvent（提交时创建一次）：HITL 恢复必须复用同一 task_id/预算/deadline
    task_event: Any = None
    #: 待批请求（status=awaiting_approval 时由 pipeline 给出）
    approval_requests: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def publish(self, event: str, **payload: Any) -> None:
        record = {"event": event, "at": _now_iso(), **payload}
        self.events.append(record)
        for queue in list(self.subscribers):
            # 队列满说明订阅方消费不过来；丢弃事件而不是阻塞主流程
            try:
                queue.put_nowait(record)
            except asyncio.QueueFull:  # pragma: no cover
                pass

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "approval_requests": self.approval_requests,
            "result": self.result,
            "event_count": len(self.events),
        }


class TaskRegistry:
    """进程内任务表。"""

    def __init__(self, *, max_events: int = 500) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._max_events = max_events

    def create(self, task_id: str, question: str) -> TaskRecord:
        record = TaskRecord(task_id=task_id, question=question)
        self._tasks[task_id] = record
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        items = sorted(self._tasks.values(), key=lambda r: r.created_at, reverse=True)
        return [r.summary() for r in items[:limit]]

    def trim(self, record: TaskRecord) -> None:
        """事件过多时截断，避免长期运行的任务占满内存。"""
        if len(record.events) > self._max_events:
            del record.events[: len(record.events) - self._max_events]
