"""黑板契约化测试：条目过冻结契约、乐观锁、订阅可见性。

黑板是任务内唯一事实源，以下三件事都是**契约语义**（`blackboard_entry.schema.json`），
不是内部实现细节：

1. 条目形状必须能被 `BlackboardEntry` 校验通过——手写近似 dict 会让契约形同虚设；
2. 覆盖既有条目走条目级版本号乐观锁：提交读到的 version，比对不符即拒绝；
3. 下游节点按订阅（直接上游）拿到条目与来源明细。
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.agents import AnalystAgent  # noqa: E402
from multiagent.contracts import (  # noqa: E402
    AgentInvocation,
    AgentResult,
    BlackboardEntry,
    EntryStatus,
    InvocationBudget,
    ResultStatus,
    SubtaskNode,
)
from multiagent.graph.pipeline import AnalysisPipeline, PipelineNodeFailed  # noqa: E402

TZ_CN = timezone(timedelta(hours=8))
FIXED_AT = datetime(2026, 10, 16, 12, 0, 0, tzinfo=TZ_CN)


class _StubReviewer:
    """reviewer 的真实实现属 P2；黑板书只依赖 `agent.version` 这一个属性。"""

    role = "reviewer"
    version = "9.9.9"


def _inv(node_id: str, role: str) -> AgentInvocation:
    return AgentInvocation(
        task_id="t-1",
        node_id=node_id,
        attempt=1,
        agent_role=role,
        trace_id="t-1",
        input={},
        budget=InvocationBudget(max_tokens=1000, timeout_seconds=60),
    )


def _result(
    node_id: str,
    *,
    outputs: dict,
    writes: list[str],
    sources: list[dict] | None = None,
) -> AgentResult:
    return AgentResult(
        task_id="t-1",
        node_id=node_id,
        attempt=1,
        status=ResultStatus.ok,
        tokens_used={"input": 1, "output": 1, "total": 2},
        duration_ms=1,
        outputs=outputs,
        sources=sources or [{"type": "sql", "ref": "sha256:abc"}],
        blackboard_writes=writes,
    )


def _analyst_entry() -> dict:
    inv = _inv("analyst_1", "analyst")
    res = _result(
        "analyst_1",
        outputs={"summary": "华东环比下降", "sql_hash": "abc"},
        writes=["analyst_1.report"],
    )
    delta = AnalysisPipeline._absorb(inv, res, AnalystAgent(), written_at=FIXED_AT)
    return delta["blackboard"]


# ---------------------------------------------------------------- 契约一致性


def test_blackboard_entry_is_contract_conformant() -> None:
    """流水线产出的条目必须能被冻结契约解析，字段一个不缺。"""
    entry = _analyst_entry()["analyst_1.report"]
    parsed = BlackboardEntry.model_validate(entry)

    assert parsed.task_id == "t-1"
    assert parsed.key == "analyst_1.report"
    assert parsed.version == 1
    assert parsed.status is EntryStatus.draft  # 新写入为 draft，核验后流转
    assert parsed.written_by.agent_role == "analyst"
    assert parsed.written_by.agent_version == AnalystAgent.version
    assert parsed.written_by.node_id == "analyst_1"
    assert parsed.written_by.attempt == 1
    assert parsed.sources and parsed.sources[0].ref == "sha256:abc"
    assert parsed.payload["summary"] == "华东环比下降"


# ---------------------------------------------------------------- 乐观锁


def test_update_succeeds_with_matching_version() -> None:
    """提交读到的 version → 版本 +1、状态流转、写入者换成核验方。"""
    current = _analyst_entry()
    rev_inv = _inv("reviewer_1", "reviewer")
    rev_res = _result(
        "reviewer_1",
        outputs={
            "verdict": "pass",
            "blackboard_updates": [
                {"key": "analyst_1.report", "expected_version": 1, "status": "validated"}
            ],
        },
        writes=["reviewer_1.review"],
    )

    delta = AnalysisPipeline._absorb(
        rev_inv, rev_res, _StubReviewer(), blackboard=current, written_at=FIXED_AT
    )
    updated = BlackboardEntry.model_validate(delta["blackboard"]["analyst_1.report"])

    assert updated.version == 2
    assert updated.status is EntryStatus.validated
    assert updated.written_by.agent_role == "reviewer"
    assert updated.written_by.node_id == "reviewer_1"
    # 状态流转不得改写原 payload（审查动作不是内容改写）
    assert updated.payload == current["analyst_1.report"]["payload"]
    # 审查方自己的条目照常新写
    assert BlackboardEntry.model_validate(
        delta["blackboard"]["reviewer_1.review"]
    ).version == 1


def test_update_with_stale_version_is_rejected() -> None:
    """提交过期版本 → 拒绝写入（可重试），绝不静默覆盖。"""
    current = _analyst_entry()
    rev_inv = _inv("reviewer_1", "reviewer")
    rev_res = _result(
        "reviewer_1",
        outputs={
            "verdict": "fail",
            "blackboard_updates": [
                {"key": "analyst_1.report", "expected_version": 5, "status": "rejected"}
            ],
        },
        writes=[],
    )

    with pytest.raises(PipelineNodeFailed) as excinfo:
        AnalysisPipeline._absorb(
            rev_inv, rev_res, _StubReviewer(), blackboard=current, written_at=FIXED_AT
        )
    assert excinfo.value.code == "BLACKBOARD_CONFLICT"
    assert excinfo.value.retryable is True
    assert "v5" in str(excinfo.value) and "v1" in str(excinfo.value)


def test_update_of_missing_key_is_rejected() -> None:
    inv = _inv("reviewer_1", "reviewer")
    res = _result(
        "reviewer_1",
        outputs={
            "blackboard_updates": [
                {"key": "analyst_9.report", "expected_version": 1, "status": "validated"}
            ]
        },
        writes=[],
    )
    with pytest.raises(PipelineNodeFailed) as excinfo:
        AnalysisPipeline._absorb(inv, res, _StubReviewer(), blackboard={}, written_at=FIXED_AT)
    assert excinfo.value.code == "BLACKBOARD_CONFLICT"
    assert excinfo.value.retryable is False


def test_rewrite_of_existing_key_is_rejected() -> None:
    """对已存在的 key 直接新写（而非走 updates）也要拒绝——两条写路径必须分清。"""
    current = _analyst_entry()
    inv = _inv("analyst_1", "analyst")
    res = _result(
        "analyst_1", outputs={"summary": "改写"}, writes=["analyst_1.report"]
    )
    with pytest.raises(PipelineNodeFailed) as excinfo:
        AnalysisPipeline._absorb(
            inv, res, AnalystAgent(), blackboard=current, written_at=FIXED_AT
        )
    assert excinfo.value.code == "BLACKBOARD_CONFLICT"
    assert "blackboard_updates" in str(excinfo.value)


def test_invalid_update_shape_is_rejected() -> None:
    inv = _inv("reviewer_1", "reviewer")
    res = _result(
        "reviewer_1",
        outputs={"blackboard_updates": [{"key": "analyst_1.report"}]},  # 缺 expected_version
        writes=[],
    )
    with pytest.raises(PipelineNodeFailed) as excinfo:
        AnalysisPipeline._absorb(
            inv, res, _StubReviewer(), blackboard=_analyst_entry(), written_at=FIXED_AT
        )
    assert excinfo.value.code == "BAD_BLACKBOARD_UPDATE"


# ---------------------------------------------------------------- 订阅可见性


def test_node_input_delivers_upstream_blackboard_and_sources() -> None:
    """节点输入里只出现**直接上游**的条目与来源，保留键不可被上游覆盖。"""
    node = SubtaskNode(
        node_id="analyst_1",
        name="取数归因",
        agent_role="analyst",
        depends_on=["retriever_1", "retriever_2"],
        acceptance=["给出至少一个渠道的下滑幅度"],
    )
    state = {
        "task_event": {"payload": {"text": "分析下滑原因"}},
        "now": "2026-10-16T12:00:00+08:00",
        "node_outputs": {
            "retriever_1": {"metric_definitions": {"revenue": {}}, "question": "被覆盖的脏键"},
            "retriever_2": {"knowledge": [{"doc_id": "return-policy"}]},
            "unrelated_9": {"summary": "无关节点"},
        },
        "blackboard": {
            "retriever_1.metrics": {"version": 1, "key": "retriever_1.metrics"},
            "retriever_2.knowledge": {"version": 1, "key": "retriever_2.knowledge"},
            "unrelated_9.report": {"version": 1, "key": "unrelated_9.report"},
        },
        "node_results": [
            {"node_id": "retriever_1", "sources": [{"type": "sql", "ref": "s1"}]},
            {"node_id": "retriever_2", "sources": [{"type": "document", "ref": "d1"}]},
            {"node_id": "unrelated_9", "sources": [{"type": "sql", "ref": "s9"}]},
        ],
    }

    inputs = AnalysisPipeline._build_node_input(node, state)

    assert set(inputs["blackboard"]) == {"retriever_1.metrics", "retriever_2.knowledge"}
    assert inputs["upstream_sources"] == {
        "retriever_1": [{"type": "sql", "ref": "s1"}],
        "retriever_2": [{"type": "document", "ref": "d1"}],
    }
    # 非上游的产物与条目都不出现
    assert "summary" not in inputs and "unrelated_9.report" not in inputs["blackboard"]
    # 保留键最后套上：上游 outputs 里同名键（question）不能覆盖真实问题
    assert inputs["question"] == "分析下滑原因"
    assert inputs["acceptance"] == ["给出至少一个渠道的下滑幅度"]
    assert inputs["node_id"] == "analyst_1"
