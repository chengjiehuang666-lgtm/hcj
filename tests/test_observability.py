"""W10 可观测：结构化日志、task_id 贯穿、Token/成本计量。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from multiagent.gateway.base import LLMMessage
from multiagent.gateway.gateway import LLMGateway
from multiagent.gateway.stub import StubProvider
from multiagent.observability import (
    build_metrics,
    configure_logging,
    log_event,
    node_context,
    set_task_id,
    summarize_calls,
)


def test_log_event_carries_task_and_node_context() -> None:
    configure_logging(level="INFO", fmt="json", force=True)
    set_task_id("task-1")
    with node_context("analyst_1", "analyst"):
        payload = log_event("model_call", route="analyst_sql", tokens=10)
    assert payload["task_id"] == "task-1"
    assert payload["node_id"] == "analyst_1"
    assert payload["agent_role"] == "analyst"
    assert payload["route"] == "analyst_sql"


def test_node_context_restores_previous_values() -> None:
    set_task_id("task-2")
    with node_context("n1", "analyst"):
        assert log_event("x")["node_id"] == "n1"
    assert log_event("x").get("node_id") is None


def test_gateway_records_task_and_agent_context() -> None:
    gateway = LLMGateway(StubProvider(name="stub", model="stub-model"))
    set_task_id("task-3")
    with node_context("planner_1", "planner"):
        asyncio.run(gateway.complete([LLMMessage.user("hi")], route="planner"))
    record = gateway.history[-1]
    assert record.task_id == "task-3"
    assert record.node_id == "planner_1"
    assert record.agent_role == "planner"
    assert record.route == "planner"


def test_summarize_calls_by_route_agent_model() -> None:
    rates = {"deepseek-v4.1-flash": 0.5}
    records = [
        {"route": "planner", "agent_role": "planner", "model": "deepseek-v4.1-flash",
         "total_tokens": 2000, "input_tokens": 1500, "output_tokens": 500, "duration_ms": 100},
        {"route": "analyst_sql", "agent_role": "analyst", "model": "deepseek-v4.1-flash",
         "total_tokens": 1000, "input_tokens": 800, "output_tokens": 200, "duration_ms": 50},
        {"route": "analyst_sql", "agent_role": "analyst", "model": "unknown-model",
         "total_tokens": 1000, "duration_ms": 50},
    ]
    summary = summarize_calls(records, rates)
    assert summary["calls"] == 3
    assert summary["total_tokens"] == 4000
    assert summary["total_cost"] == 1.5
    assert summary["by_route"]["planner"]["cost"] == 1.0
    assert summary["by_route"]["analyst_sql"]["total_tokens"] == 2000
    assert summary["by_agent"]["analyst"]["calls"] == 2
    assert summary["by_model"]["unknown-model"]["cost"] == 0.0


def test_build_metrics_aggregates_tasks() -> None:
    records = [
        SimpleNamespace(
            status="ok",
            result={
                "total_tokens": 100,
                "cost": 0.5,
                "duration_ms": 1000,
                "nodes": [
                    {"agent_role": "analyst", "tokens": 60},
                    {"agent_role": "reviewer", "tokens": 40},
                ],
                "tokens_by_route": {"analyst_sql": {"total_tokens": 60}},
                "cost_by_route": {"analyst_sql": {"cost": 0.3}},
            },
        ),
        SimpleNamespace(
            status="failed",
            result={"total_tokens": 20, "cost": 0.1, "duration_ms": 500},
        ),
    ]
    metrics = build_metrics(records)
    assert metrics["tasks"]["total"] == 2
    assert metrics["tasks"]["by_status"] == {"ok": 1, "failed": 1}
    assert metrics["tasks"]["avg_duration_ms"] == 750
    assert metrics["tokens"]["total"] == 120
    assert metrics["tokens"]["by_agent"] == {"analyst": 60, "reviewer": 40}
    assert metrics["cost"]["total"] == 0.6
    assert metrics["cost"]["by_route"]["analyst_sql"] == 0.3
