"""契约一致性测试：Pydantic 实现的输出必须能被**冻结的 JSON Schema** 接受。

这是「契约先行」的守卫：一旦 `src/multiagent/contracts/` 与
`架构设想/接口契约/*.schema.json` 发生漂移，本测试立即失败。

运行方式（本机未装 pytest 时）：
    python tests/test_contracts_conformance.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsonschema import Draft202012Validator, FormatChecker  # noqa: E402

from multiagent.contracts import (  # noqa: E402
    AgentError,
    AgentInvocation,
    AgentResult,
    AgentRole,
    BlackboardEntry,
    Budget,
    EntryStatus,
    InvocationBudget,
    Payload,
    ResultStatus,
    Sensitivity,
    Source,
    SourceChannel,
    SourceRef,
    SourceType,
    SubtaskDAG,
    SubtaskNode,
    TaskEvent,
    TokenUsage,
    User,
    UserRole,
    WrittenBy,
    to_wire,
)

SCHEMA_DIR = ROOT / "架构设想" / "接口契约"
NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone(timedelta(hours=8)))
TASK_ID = "3f2a1b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b"
WIRE_TASK_ID = str(uuid.UUID(TASK_ID))


def _validator(filename: str, pointer: str | None = None) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))
    if pointer:
        for part in pointer.strip("/").split("/"):
            schema = schema[part]
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _assert_valid(validator: Draft202012Validator, instance: dict, label: str) -> None:
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    assert not errors, f"{label} 违反契约:\n" + "\n".join(
        f"  - {'/'.join(map(str, e.path))}: {e.message}" for e in errors
    )


# --------------------------------------------------------------------------- 样本


def make_task_event() -> TaskEvent:
    return TaskEvent(
        task_id=uuid.UUID(TASK_ID),
        source=Source(channel=SourceChannel.rest_api),
        submitted_at=NOW,
        user=User(user_id="u_1001", roles=[UserRole.business_user]),
        payload=Payload(text="分析上月华东区销售额下滑原因"),
        sensitivity=Sensitivity.internal,
        budget=Budget(deadline=NOW + timedelta(minutes=5)),
    )


def make_dag() -> SubtaskDAG:
    return SubtaskDAG(
        task_id=WIRE_TASK_ID,
        goal="分析上月华东区销售额下滑原因",
        created_at=NOW,
        nodes=[
            SubtaskNode(
                node_id="retriever_1",
                name="检索指标口径与历史结论",
                agent_role=AgentRole.retriever,
                acceptance=["返回口径定义", "每条结论带来源"],
                outputs={"definition": {"type": "string"}},
            ),
            SubtaskNode(
                node_id="analyst_1",
                name="取数并做归因分析",
                agent_role=AgentRole.analyst,
                depends_on=["retriever_1"],
                acceptance=["给出下滑归因", "附可回溯 SQL"],
                outputs={"report": {"type": "string"}},
            ),
            SubtaskNode(
                node_id="reviewer_1",
                name="交叉校验",
                agent_role=AgentRole.reviewer,
                depends_on=["analyst_1"],
                acceptance=["逐条核对上游验收标准"],
                requires_human_approval=True,
                outputs={"verdict": {"type": "string"}},
            ),
        ],
    )


def make_entry() -> BlackboardEntry:
    return BlackboardEntry(
        task_id=WIRE_TASK_ID,
        key="analyst_1.report",
        version=1,
        payload={"summary": "下滑主因是渠道 A 退货率上升"},
        written_by=WrittenBy(
            agent_role="analyst",
            agent_version="analyst@0.1.0+prompt-v1",
            node_id="analyst_1",
            attempt=1,
        ),
        written_at=NOW,
        status=EntryStatus.validated,
        sources=[SourceRef(type=SourceType.sql, ref="sha256:abc123", timestamp=NOW)],
        confidence=0.82,
    )


def make_invocation() -> AgentInvocation:
    return AgentInvocation(
        task_id=WIRE_TASK_ID,
        node_id="analyst_1",
        attempt=1,
        agent_role="analyst",
        trace_id=WIRE_TASK_ID,
        input={"question": "上月华东区销售额为何下滑"},
        budget=InvocationBudget(max_tokens=50_000, timeout_seconds=300),
        visible_blackboard_keys=["retriever_1.definition"],
    )


def make_result_ok() -> AgentResult:
    return AgentResult(
        task_id=WIRE_TASK_ID,
        node_id="analyst_1",
        attempt=1,
        status=ResultStatus.ok,
        tokens_used=TokenUsage(input=1200, output=800, total=2000),
        duration_ms=4300,
        outputs={"report": "下滑归因：..."},
        confidence=0.9,
        sources=[SourceRef(type=SourceType.sql, ref="sha256:abc123")],
    )


# ----------------------------------------------------------------------- 正向测试


def test_task_event_conforms_to_schema() -> None:
    _assert_valid(
        _validator("task_event.schema.json"), to_wire(make_task_event()), "TaskEvent"
    )


def test_subtask_dag_conforms_to_schema() -> None:
    _assert_valid(
        _validator("subtask_dag.schema.json"), to_wire(make_dag()), "SubtaskDAG"
    )


def test_blackboard_entry_conforms_to_schema() -> None:
    _assert_valid(
        _validator("blackboard_entry.schema.json"), to_wire(make_entry()), "BlackboardEntry"
    )


def test_agent_invocation_conforms_to_schema() -> None:
    _assert_valid(
        _validator("agent_io.schema.json", "/$defs/AgentInvocation"),
        to_wire(make_invocation()),
        "AgentInvocation",
    )


def test_planner_invocation_conforms_to_schema() -> None:
    """Planner 是图外产出者，但它确实通过 AgentInvocation 调用——
    枚举必须含 planner。这是 W7 修正的长期契约漂移（此前每次规划都发出
    不合冻结 Schema 的消息，而契约守卫从未以 planner 取样）。"""
    invocation = make_invocation().model_copy(
        update={"node_id": "planner_1", "agent_role": "planner"}
    )
    _assert_valid(
        _validator("agent_io.schema.json", "/$defs/AgentInvocation"),
        to_wire(invocation),
        "AgentInvocation(planner)",
    )


def test_planner_blackboard_writer_conforms_to_schema() -> None:
    entry = make_entry().model_copy(
        update={
            "written_by": WrittenBy(
                agent_role="planner", agent_version="0.2.0", node_id="planner_1", attempt=1
            )
        }
    )
    _assert_valid(
        _validator("blackboard_entry.schema.json"), to_wire(entry), "BlackboardEntry(planner)"
    )


def test_agent_result_conforms_to_schema() -> None:
    _assert_valid(
        _validator("agent_io.schema.json", "/$defs/AgentResult"),
        to_wire(make_result_ok()),
        "AgentResult",
    )


def test_failed_result_conforms_to_schema() -> None:
    result = AgentResult(
        task_id=WIRE_TASK_ID,
        node_id="analyst_1",
        attempt=2,
        status=ResultStatus.failed,
        tokens_used=TokenUsage(input=100, output=0, total=100),
        duration_ms=900,
        error=AgentError(code="SQL_ERROR", message="relation does not exist", retryable=False),
    )
    _assert_valid(
        _validator("agent_io.schema.json", "/$defs/AgentResult"), to_wire(result), "AgentResult(failed)"
    )


# ----------------------------------------------------------------------- 反向测试


def test_unknown_field_rejected() -> None:
    payload = to_wire(make_task_event())
    payload["unexpected_field"] = 1
    assert _validator("task_event.schema.json").iter_errors(payload), "契约应拒绝未声明字段"
    try:
        TaskEvent(**{**payload, "unexpected_field": 1})
    except Exception:
        return
    raise AssertionError("Pydantic 模型应拒绝未声明字段（extra=forbid）")


def test_dag_cycle_rejected() -> None:
    def node(nid: str, deps: list[str]) -> SubtaskNode:
        return SubtaskNode(
            node_id=nid, name=nid, agent_role=AgentRole.analyst, depends_on=deps,
            acceptance=["x"],
        )

    try:
        SubtaskDAG(
            task_id=WIRE_TASK_ID, goal="g", created_at=NOW,
            nodes=[node("a_1", ["b_1"]), node("b_1", ["a_1"])],
        )
    except ValueError as exc:
        assert "循环依赖" in str(exc), f"错误信息应指出循环依赖，实际: {exc}"
        return
    raise AssertionError("循环依赖的 DAG 应被拒绝")


def test_dag_dangling_dependency_rejected() -> None:
    try:
        SubtaskDAG(
            task_id=WIRE_TASK_ID, goal="g", created_at=NOW,
            nodes=[
                SubtaskNode(node_id="analyst_1", name="a", agent_role=AgentRole.analyst,
                            depends_on=["ghost_1"], acceptance=["x"])
            ],
        )
    except ValueError as exc:
        assert "不存在" in str(exc)
        return
    raise AssertionError("引用不存在节点的 DAG 应被拒绝")


def test_dag_duplicate_node_id_rejected() -> None:
    dup = SubtaskNode(node_id="a_1", name="a", agent_role=AgentRole.analyst, acceptance=["x"])
    try:
        SubtaskDAG(task_id=WIRE_TASK_ID, goal="g", created_at=NOW, nodes=[dup, dup])
    except ValueError as exc:
        assert "重复" in str(exc)
        return
    raise AssertionError("node_id 重复的 DAG 应被拒绝")


def test_agent_result_ok_requires_sources() -> None:
    try:
        AgentResult(
            task_id=WIRE_TASK_ID, node_id="analyst_1", attempt=1, status=ResultStatus.ok,
            tokens_used=TokenUsage(input=1, output=1, total=2), duration_ms=10,
            outputs={"report": "x"},
        )
    except ValueError as exc:
        assert "sources" in str(exc)
        return
    raise AssertionError("status=ok 但缺 sources 时应被拒绝（溯源强制）")


def test_agent_result_failed_requires_error() -> None:
    try:
        AgentResult(
            task_id=WIRE_TASK_ID, node_id="analyst_1", attempt=1, status=ResultStatus.failed,
            tokens_used=TokenUsage(input=1, output=0, total=1), duration_ms=10,
        )
    except ValueError as exc:
        assert "error" in str(exc)
        return
    raise AssertionError("status=failed 但缺 error 时应被拒绝")


def test_budget_cannot_be_widened() -> None:
    budget = Budget(max_steps=30, max_tokens=200_000, max_retries=2, deadline=NOW)
    tightened = budget.tightened(max_tokens=50_000)
    assert tightened.max_tokens == 50_000
    try:
        budget.tightened(max_tokens=999_999)
    except ValueError as exc:
        assert "收紧" in str(exc)
        return
    raise AssertionError("放宽预算应被拒绝（契约要求预算只能收紧）")


def test_ready_nodes_respects_dependencies() -> None:
    dag = make_dag()
    assert [n.node_id for n in dag.ready_nodes(set())] == ["retriever_1"]
    assert [n.node_id for n in dag.ready_nodes({"retriever_1"})] == ["analyst_1"]
    assert [n.node_id for n in dag.ready_nodes({"retriever_1", "analyst_1"})] == ["reviewer_1"]


def test_entry_next_version_increments() -> None:
    entry = make_entry()
    newer = entry.next_version({"summary": "修正后的结论"})
    assert newer.version == entry.version + 1
    assert newer.key == entry.key


# ------------------------------------------------------------------------- 运行器


def _run() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
