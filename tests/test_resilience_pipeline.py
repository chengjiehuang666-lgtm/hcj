"""W7：有界动态重规划（run() 层）——真实 checkpointer + 假 Agent。

验证：
1. 节点重试耗尽（可重试失败）→ 触发一次重规划，Planner 拿到上一版 DAG 与失败原因；
2. 新 DAG 在**新 thread** 从头执行，revision 由代码递增（不是模型自填）；
3. `max_replans=0` 时不重规划；`needs_human` 直接终止、不重规划；
4. 重规划次数用尽后仍失败 → 以最后一次失败终止（revision=2）。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.agents import BaseAgent  # noqa: E402
from multiagent.agents.base import _Usage  # noqa: E402
from multiagent.asyncutil import run_async  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.contracts import (  # noqa: E402
    AgentInvocation,
    AgentResult,
    SourceRef,
    SourceType,
)
from multiagent.graph.pipeline import AnalysisPipeline, new_task_event  # noqa: E402

settings = Settings(_env_file=ROOT / ".env")
_test_settings = settings.model_copy(
    update={"memory_sink_enabled": False, "max_replans": 1}
)


def _skip_if_db_unreachable() -> None:
    import psycopg

    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=5):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可达: {exc}")


def _dag_json(role: str, node_id: str) -> dict:
    return {
        "task_id": "t",
        "goal": "g",
        "created_at": "2026-10-17T12:00:00+08:00",
        "nodes": [
            {
                "node_id": node_id,
                "name": node_id,
                "agent_role": role,
                "depends_on": [],
                "inputs": {},
                "outputs": {},
                "acceptance": ["x"],
            }
        ],
    }


class _ScriptedPlanner(BaseAgent):
    """首版给 analyst；重规划时给 retriever（模拟「换一种拆解」）。"""

    role = "planner"
    version = "test"

    def __init__(self) -> None:
        self.inputs: list[dict] = []

    async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
        self.inputs.append(dict(inv.input))
        if inv.input.get("previous_dag"):
            dag = _dag_json("retriever", "retriever_1")
        else:
            dag = _dag_json("analyst", "analyst_1")
        return self.ok(
            inv,
            outputs={"dag": dag, "node_count": 1},
            sources=[SourceRef(type=SourceType.model_inference, ref="test")],
            usage=_Usage(),
            duration_ms=1,
        )


class _FailingAnalyst(BaseAgent):
    role = "analyst"
    version = "test"

    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
        self.calls += 1
        return self.failed(
            inv,
            code="PROVIDER_UNAVAILABLE",
            message="模拟供应商持续抖动",
            retryable=True,
            usage=_Usage(),
            duration_ms=1,
        )


class _NeedsReplanAnalyst(BaseAgent):
    role = "analyst"
    version = "test"

    async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
        return AgentResult(
            task_id=inv.task_id,
            node_id=inv.node_id,
            attempt=inv.attempt,
            status="needs_replan",
            tokens_used={"input": 1, "output": 1, "total": 2},
            duration_ms=1,
        )


class _NeedsHumanAnalyst(BaseAgent):
    role = "analyst"
    version = "test"

    async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
        return AgentResult(
            task_id=inv.task_id,
            node_id=inv.node_id,
            attempt=inv.attempt,
            status="needs_human",
            tokens_used={"input": 1, "output": 1, "total": 2},
            duration_ms=1,
        )


class _WorkingRetriever(BaseAgent):
    role = "retriever"
    version = "test"

    async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
        return self.ok(
            inv,
            outputs={"metrics": "ok", "summary": "重规划后的结论"},
            sources=[SourceRef(type=SourceType.model_inference, ref="test")],
            usage=_Usage(),
            duration_ms=1,
            blackboard_writes=["retriever_1.metrics"],
        )


def _run_case(monkeypatch, analyst_agent, *, settings_override=None):
    _skip_if_db_unreachable()
    from multiagent.graph import pipeline as pipeline_module

    planner = _ScriptedPlanner()
    registry = {
        "planner": planner,
        "analyst": analyst_agent,
        "retriever": _WorkingRetriever(),
    }
    monkeypatch.setattr(pipeline_module, "build_agent_registry", lambda: registry)
    cfg = settings_override or _test_settings
    pipeline = AnalysisPipeline(cfg)
    event = new_task_event("分析一下销售额", settings=cfg)
    outcome = run_async(pipeline.run(event))
    return outcome, planner


def test_retryable_failure_triggers_one_replan(monkeypatch) -> None:
    analyst = _FailingAnalyst()
    outcome, planner = _run_case(monkeypatch, analyst)

    assert outcome.status == "ok", outcome.error
    assert outcome.revision == 2
    assert outcome.dag["revision"] == 2, "revision 由代码递增"
    assert outcome.dag["nodes"][0]["node_id"] == "retriever_1", "执行的是新 DAG"
    # Planner 第二次调用拿到了失败上下文
    assert len(planner.inputs) == 2
    assert "failure_reason" in planner.inputs[1]
    assert "PROVIDER_UNAVAILABLE" in planner.inputs[1]["failure_reason"]
    assert planner.inputs[1]["revision"] == 2
    # 节点级重试仍然先发生（1 次初始 + 2 次重试），之后才重规划
    assert analyst.calls == 3


def test_needs_replan_status_triggers_replan(monkeypatch) -> None:
    outcome, planner = _run_case(monkeypatch, _NeedsReplanAnalyst())

    assert outcome.status == "ok", outcome.error
    assert outcome.revision == 2
    assert len(planner.inputs) == 2


def test_max_replans_zero_disables_replan(monkeypatch) -> None:
    cfg = settings.model_copy(update={"memory_sink_enabled": False, "max_replans": 0})
    analyst = _FailingAnalyst()
    outcome, planner = _run_case(monkeypatch, analyst, settings_override=cfg)

    assert outcome.status == "failed"
    assert outcome.error_code == "PROVIDER_UNAVAILABLE"
    assert outcome.revision == 1
    assert len(planner.inputs) == 1, "max_replans=0 不得重规划"


def test_needs_human_does_not_replan(monkeypatch) -> None:
    outcome, planner = _run_case(monkeypatch, _NeedsHumanAnalyst())

    assert outcome.status == "failed"
    assert outcome.error_code == "NEEDS_HUMAN"
    assert outcome.revision == 1
    assert len(planner.inputs) == 1, "needs_human 是转人工终止，不是重规划"


def test_exhausted_replans_end_with_last_failure(monkeypatch) -> None:
    """重规划后仍然失败：不再重规划（上限 1），以最后失败终止、revision=2。"""

    class _AlwaysFailingRetriever(BaseAgent):
        role = "retriever"
        version = "test"

        async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
            return self.failed(
                inv,
                code="PROVIDER_UNAVAILABLE",
                message="仍然失败",
                retryable=True,
                usage=_Usage(),
                duration_ms=1,
            )

    _skip_if_db_unreachable()
    from multiagent.graph import pipeline as pipeline_module

    planner = _ScriptedPlanner()
    registry = {
        "planner": planner,
        "analyst": _FailingAnalyst(),
        "retriever": _AlwaysFailingRetriever(),
    }
    monkeypatch.setattr(pipeline_module, "build_agent_registry", lambda: registry)
    pipeline = AnalysisPipeline(_test_settings)
    event = new_task_event("分析一下销售额", settings=_test_settings)
    outcome = run_async(pipeline.run(event))

    assert outcome.status == "failed"
    assert outcome.revision == 2, "用满 1 次重规划后终止"
    assert len(planner.inputs) == 2
