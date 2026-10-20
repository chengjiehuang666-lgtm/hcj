"""可观测（W10）：结构化日志、task_id 贯穿、Token/成本计量。

- **task_id 贯穿**：用 `ContextVar` 在任务/节点边界写入，网关、Agent、工具
  的日志与调用记录自动带上，不需要层层传参；asyncio 下 ContextVar 是任务隔离的。
- **结构化日志**：JSON 一行一条，字段固定（ts/level/event/task_id/node_id/
  agent_role + 事件字段）。不记录原始 prompt/输出/密钥，只记长度、哈希、
  token、耗时——**日志本身也不能成为泄露渠道**。
- **成本计量**：按 `CallRecord` 的 model + tokens 乘以配置单价，汇总到
  调用点 / Agent / 模型三个维度；单价是本地端点积分，不是货币。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)
node_id_var: ContextVar[str | None] = ContextVar("node_id", default=None)
agent_role_var: ContextVar[str | None] = ContextVar("agent_role", default=None)

_LOGGER_NAME = "multiagent"
_CONFIGURED = False


def set_task_id(task_id: str | None) -> None:
    task_id_var.set(str(task_id) if task_id is not None else None)


def current_context() -> dict[str, Any]:
    return {
        "task_id": task_id_var.get(),
        "node_id": node_id_var.get(),
        "agent_role": agent_role_var.get(),
    }


@contextmanager
def node_context(node_id: str, agent_role: str) -> Iterator[None]:
    """节点执行边界：写入 node_id/agent_role，退出时恢复。"""
    node_token = node_id_var.set(str(node_id))
    role_token = agent_role_var.set(str(agent_role))
    try:
        yield
    finally:
        agent_role_var.reset(role_token)
        node_id_var.reset(node_token)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "event_payload", None)
        if payload is None:
            payload = {"event": record.getMessage()}
        body = {
            "ts": datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            **payload,
        }
        return json.dumps(body, ensure_ascii=False, default=str)


def configure_logging(
    settings: Any | None = None,
    *,
    level: str | None = None,
    fmt: str | None = None,
    force: bool = False,
) -> None:
    """配置 `multiagent` 命名空间的日志；重复调用不会叠加 handler。"""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    log_level = (level or getattr(settings, "log_level", None) or "INFO").upper()
    log_format = fmt or getattr(settings, "log_format", None) or "json"
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False
    _CONFIGURED = True


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> dict[str, Any]:
    """写一条结构化日志，并返回实际写入的字段（便于测试/复用）。"""
    payload = {"event": event, **{k: v for k, v in fields.items() if v is not None}}
    for key, value in current_context().items():
        if value is not None:
            payload.setdefault(key, value)
    logging.getLogger(_LOGGER_NAME).log(level, "", extra={"event_payload": payload})
    return payload


# ---------------------------------------------------------------- 成本计量


def _rate_for(model: str | None, rates: dict[str, float]) -> float:
    if not model:
        return 0.0
    if model in rates:
        return float(rates[model])
    # 端点的 model 字段可能带命名空间（cn:...），做一次包含匹配
    for name, rate in rates.items():
        if name and name in model:
            return float(rate)
    return 0.0


def call_cost(record: dict[str, Any], rates: dict[str, float]) -> float:
    return float(record.get("total_tokens") or 0) / 1000.0 * _rate_for(
        record.get("model"), rates
    )


def summarize_calls(
    records: Iterable[dict[str, Any]], rates: dict[str, float]
) -> dict[str, Any]:
    """按调用点 / Agent / 模型汇总 token 与成本。"""
    by_route: dict[str, dict[str, Any]] = {}
    by_agent: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    total_cost = 0.0
    calls = 0

    def bucket(store: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        return store.setdefault(
            key,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "total_tokens": 0, "duration_ms": 0, "cost": 0.0},
        )

    for record in records:
        calls += 1
        tokens = int(record.get("total_tokens") or 0)
        cost = call_cost(record, rates)
        total_tokens += tokens
        total_cost += cost
        for store, key in (
            (by_route, str(record.get("route") or "(未标注)")),
            (by_agent, str(record.get("agent_role") or "(未标注)")),
            (by_model, str(record.get("model") or "(未知)")),
        ):
            item = bucket(store, key)
            item["calls"] += 1
            item["input_tokens"] += int(record.get("input_tokens") or 0)
            item["output_tokens"] += int(record.get("output_tokens") or 0)
            item["total_tokens"] += tokens
            item["duration_ms"] += int(record.get("duration_ms") or 0)
            item["cost"] = round(item["cost"] + cost, 6)

    return {
        "calls": calls,
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6),
        "by_route": by_route,
        "by_agent": by_agent,
        "by_model": by_model,
    }


def build_metrics(records: Iterable[Any]) -> dict[str, Any]:
    """从进程内任务记录汇总指标（重启清零，单进程口径）。"""
    statuses: dict[str, int] = {}
    tokens_by_agent: dict[str, int] = {}
    tokens_by_route: dict[str, int] = {}
    cost_by_route: dict[str, float] = {}
    total_tokens = 0
    total_cost = 0.0
    finished = 0
    duration_total = 0

    for record in records:
        statuses[record.status] = statuses.get(record.status, 0) + 1
        result = record.result or {}
        total_tokens += int(result.get("total_tokens") or 0)
        total_cost += float(result.get("cost") or 0.0)
        if record.status in ("ok", "failed", "rejected"):
            finished += 1
            duration_total += int(result.get("duration_ms") or 0)
        for node in result.get("nodes") or []:
            role = str(node.get("agent_role") or "unknown")
            tokens_by_agent[role] = tokens_by_agent.get(role, 0) + int(
                node.get("tokens") or 0
            )
        for route, stats in (result.get("tokens_by_route") or {}).items():
            tokens_by_route[route] = tokens_by_route.get(route, 0) + int(
                stats.get("total_tokens") or 0
            )
        for route, stats in (result.get("cost_by_route") or {}).items():
            cost_by_route[route] = round(
                cost_by_route.get(route, 0.0) + float(stats.get("cost") or 0.0), 6
            )

    return {
        "tasks": {
            "total": sum(statuses.values()),
            "finished": finished,
            "by_status": statuses,
            "avg_duration_ms": int(duration_total / finished) if finished else 0,
        },
        "tokens": {"total": total_tokens, "by_agent": tokens_by_agent,
                   "by_route": tokens_by_route},
        "cost": {"total": round(total_cost, 6), "by_route": cost_by_route},
    }
