"""W7：HITL 端到端（run() 层）——真实 PostgreSQL checkpointer + 假 Agent。

单测（引擎层）已证明审批门的 interrupt/恢复机制；这里验证 `run()` 的三件事：
1. 暂停时返回 `awaiting_approval` 并带回已完成节点的部分结果；
2. 恢复必须用**同一 thread_id**（= TaskEvent.task_id）继续，而不是重新规划；
3. 驳回映射为独立终态 `rejected`（不是 failed），且不沉淀记忆。

数据库不可达时跳过（与其它 DB 测试同一约定）。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.agents import BaseAgent  # noqa: E402
from multiagent.asyncutil import run_async  # noqa: E402
from multiagent.agents.base import _Usage  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.contracts import (  # noqa: E402
    AgentInvocation,
    AgentResult,
    SourceRef,
    SourceType,
)
from multiagent.graph.pipeline import AnalysisPipeline, new_task_event  # noqa: E402

settings = Settings(_env_file=ROOT / ".env")
# 本测试不关心记忆沉淀：关掉它，避免加载真实 embedding 模型
_test_settings = settings.model_copy(update={"memory_sink_enabled": False})


def _skip_if_db_unreachable() -> None:
    import psycopg

    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=5):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可达: {exc}")


class _FakePlanner(BaseAgent):
    role = "planner"
    version = "test"

    async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
        dag = {
            "task_id": str(inv.task_id),
            "goal": "写一份对外公告",
            "created_at": "2026-10-17T12:00:00+08:00",
            "nodes": [
                {
                    "node_id": "writer_1",
                    "name": "写公告",
                    "agent_role": "writer",
                    "depends_on": [],
                    "inputs": {},
                    "outputs": {},
                    "acceptance": ["包含标题"],
                    "requires_human_approval": True,
                },
                {
                    "node_id": "reviewer_1",
                    "name": "审核",
                    "agent_role": "reviewer",
                    "depends_on": ["writer_1"],
                    "inputs": {},
                    "outputs": {},
                    "acceptance": ["核对标题"],
                },
            ],
        }
        return self.ok(
            inv,
            outputs={"dag": dag, "node_count": len(dag["nodes"])},
            sources=[SourceRef(type=SourceType.model_inference, ref="test")],
            usage=_Usage(),
            duration_ms=1,
        )


class _FakeWriter(BaseAgent):
    role = "writer"
    version = "test"

    async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
        return self.ok(
            inv,
            outputs={"document_markdown": "# 公告\n\n正文", "summary": "公告正文"},
            sources=[SourceRef(type=SourceType.model_inference, ref="test")],
            usage=_Usage(),
            duration_ms=1,
            blackboard_writes=["writer_1.document"],
        )


class _FakeReviewer(BaseAgent):
    role = "reviewer"
    version = "test"

    async def invoke(self, inv: AgentInvocation, ctx) -> AgentResult:
        return self.ok(
            inv,
            outputs={"verdict": "pass", "issue_count": 0},
            sources=[SourceRef(type=SourceType.model_inference, ref="test")],
            usage=_Usage(),
            duration_ms=1,
            blackboard_writes=["reviewer_1.review"],
        )


def _fake_registry() -> dict[str, BaseAgent]:
    return {
        "planner": _FakePlanner(),
        "writer": _FakeWriter(),
        "reviewer": _FakeReviewer(),
    }


def test_run_pauses_for_approval_and_resumes(monkeypatch) -> None:
    _skip_if_db_unreachable()
    from multiagent.graph import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "build_agent_registry", _fake_registry)
    pipeline = AnalysisPipeline(_test_settings)
    event = new_task_event("写一份对外公告并发送给客户", settings=_test_settings)

    first = run_async(pipeline.run(event))
    assert first.status == "awaiting_approval"
    assert first.approval_requests, "暂停时必须给出待批请求"
    payload = first.approval_requests[0]["value"]
    assert payload["node_id"] == "writer_1"
    assert payload["outputs"]["summary"] == "公告正文", "审批载荷要带被审节点的产物"
    # 被审节点（writer）已完成，它的部分结果要带回（不是空 outcome）
    assert any(r["node_id"] == "writer_1" for r in first.node_results)
    assert not any(r["agent_role"] == "reviewer" for r in first.node_results), (
        "审批未决时下游不得执行"
    )

    # 恢复：同一 event（同一 thread_id）
    second = run_async(
        pipeline.run(event, resume=True, resume_value={"decision": "approved"})
    )
    assert second.status == "ok"
    assert second.approvals["writer_1"]["decision"] == "approved"
    assert any(r["agent_role"] == "reviewer" for r in second.node_results), (
        "批准后下游应执行"
    )


def test_run_maps_rejection_to_rejected_status(monkeypatch) -> None:
    _skip_if_db_unreachable()
    from multiagent.graph import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "build_agent_registry", _fake_registry)
    pipeline = AnalysisPipeline(_test_settings)
    event = new_task_event("写一份对外公告并发送给客户", settings=_test_settings)

    first = run_async(pipeline.run(event))
    assert first.status == "awaiting_approval"
    second = run_async(
        pipeline.run(event, resume=True, resume_value={"decision": "rejected", "comment": "不发"})
    )
    assert second.status == "rejected", "驳回是独立终态，不是 failed"
    assert "人工驳回" in (second.error or "")
    assert not any(r["agent_role"] == "reviewer" for r in second.node_results)


def test_load_saved_task_reads_event_from_checkpoint(monkeypatch) -> None:
    """进程重启后的审批恢复依赖它：从 checkpoint 读回原始 TaskEvent。

    W7 实现里这里有个致命参数错误（max_size=2 < 默认 min_size=4 → 构造抛
    ValueError，被上层 except 吞掉后表现为 404）；本测试钉住修复。
    """
    _skip_if_db_unreachable()
    from multiagent.graph import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "build_agent_registry", _fake_registry)
    pipeline = AnalysisPipeline(_test_settings)
    question = "写一份对外公告并发送给客户"
    event = new_task_event(question, settings=_test_settings)

    first = run_async(pipeline.run(event))
    assert first.status == "awaiting_approval"

    loaded = run_async(pipeline.load_saved_task(str(event.task_id)))
    assert loaded is not None, "必须能从 checkpoint 读回任务事件"
    assert loaded.payload.text == question
    assert str(loaded.task_id) == str(event.task_id)

    assert run_async(pipeline.load_saved_task("no-such-thread")) is None
