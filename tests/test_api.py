"""REST + SSE 接口测试。

用假流水线注入，因此**不调用模型、不烧 token、结果确定**，
同时把接口契约（状态码、字段、事件序列、错误语义）钉死。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from multiagent.api import create_app  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.graph.pipeline import PipelineOutcome  # noqa: E402

FAKE_RESULT_OUTPUTS = {
    "summary": "华东区 9 月销售额下滑 20.85%",
    "key_findings": ["线上数码环比下降 37.6%"],
    "insufficient_data": False,
    "sql": "SELECT 1",
    "sql_hash": "deadbeefdeadbeef",
    "sql_attempts": 1,
    "row_count": 2,
}


class FakePipeline:
    """模拟流水线：发出与真实链路相同的节点事件，返回构造好的 outcome。

    `with_review=True` 时追加 W6 的 reviewer 节点与黑板变更事件，
    用于断言 SSE 与结果里能观察到评审链路（不依赖真实模型）。
    """

    def __init__(
        self,
        settings,
        on_event=None,
        *,
        should_fail: bool = False,
        with_review: bool = False,
    ) -> None:
        self._on_event = on_event
        self._should_fail = should_fail
        self._with_review = with_review

    async def run(self, event):
        if self._on_event:
            # 与真实链路一致：planner 在图外先跑，DAG 有两个并行检索 + 一个分析
            for node_id, role in (
                ("planner_1", "planner"),
                ("retriever_1", "retriever"),
                ("retriever_2", "retriever"),
                ("analyst_1", "analyst"),
            ):
                await self._on_event(
                    "node_started", {"node_id": node_id, "agent_role": role}
                )
                await self._on_event(
                    "node_finished",
                    {"node_id": node_id, "agent_role": role, "status": "ok", "tokens": 100},
                )
            if self._with_review:
                await self._on_event(
                    "node_started", {"node_id": "reviewer_1", "agent_role": "reviewer"}
                )
                await self._on_event(
                    "blackboard_written",
                    {
                        "key": "analyst_1.report",
                        "version": 2,
                        "status": "validated",
                        "written_by": {
                            "agent_role": "reviewer",
                            "agent_version": "0.1.0",
                            "node_id": "reviewer_1",
                            "attempt": 1,
                        },
                        "node_id": "reviewer_1",
                        "agent_role": "reviewer",
                    },
                )
                await self._on_event(
                    "node_finished",
                    {"node_id": "reviewer_1", "agent_role": "reviewer", "status": "ok", "tokens": 50},
                )
        if self._should_fail:
            raise RuntimeError("模拟流水线崩溃")
        node_results = [
            {
                "node_id": "analyst_1",
                "agent_role": "analyst",
                "status": "ok",
                "duration_ms": 1000,
                "tokens_used": {"input": 100, "output": 221, "total": 321},
                "outputs": FAKE_RESULT_OUTPUTS,
            }
        ]
        if self._with_review:
            node_results.append(
                {
                    "node_id": "reviewer_1",
                    "agent_role": "reviewer",
                    "status": "ok",
                    "duration_ms": 500,
                    "tokens_used": {"input": 40, "output": 30, "total": 70},
                    "outputs": {
                        "verdict": "pass",
                        "issue_count": 0,
                        "issues": [],
                        "checks": [
                            {"criterion": "结论有 SQL 支撑", "result": "met", "evidence": "x"}
                        ],
                    },
                }
            )
        return PipelineOutcome(
            task_id=str(event.task_id),
            status="ok",
            summary=FAKE_RESULT_OUTPUTS["summary"],
            total_tokens=321,
            duration_ms=1234,
            node_results=node_results,
            call_records=[
                {"route": "planner", "provider": "fake", "ok": True,
                 "input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "duration_ms": 100}
            ],
        )


def _client(*, should_fail: bool = False) -> TestClient:
    app = create_app(
        settings=Settings(_env_file=None),
        pipeline_factory=lambda s, on_event: FakePipeline(s, on_event, should_fail=should_fail),
    )
    return TestClient(app)


def _wait_finished(client: TestClient, task_id: str, tries: int = 50) -> dict:
    for _ in range(tries):
        body = client.get(f"/tasks/{task_id}").json()
        if body["status"] in ("ok", "failed"):
            return body
    raise AssertionError(f"任务未在预期内结束: {body}")


# ------------------------------------------------------------------ 健康检查


def test_health_reports_each_dependency_separately() -> None:
    """整体 ok/fail 会掩盖「哪一项挂了」，必须分项报告。"""
    body = _client().get("/health").json()
    assert set(body["dependencies"]) == {"postgres", "redis", "llm_gateway"}
    for name, dep in body["dependencies"].items():
        assert dep["status"] in ("ok", "fail"), (name, dep)
        assert "detail" in dep
    assert body["status"] in ("ok", "degraded")
    assert "routing" in body


# ------------------------------------------------------------------ 提交任务


def test_submit_returns_202_and_task_id() -> None:
    response = _client().post("/tasks", json={"question": "华东区销售额如何？"})
    assert response.status_code == 202
    body = response.json()
    assert body["task_id"]
    assert body["poll_url"] == f"/tasks/{body['task_id']}"
    assert body["events_url"] == f"/tasks/{body['task_id']}/events"


def test_empty_question_is_rejected() -> None:
    assert _client().post("/tasks", json={"question": ""}).status_code == 422


def test_task_completes_with_structured_result() -> None:
    client = _client()
    task_id = client.post("/tasks", json={"question": "q"}).json()["task_id"]
    body = _wait_finished(client, task_id)

    assert body["status"] == "ok"
    result = body["result"]
    assert result["summary"] == FAKE_RESULT_OUTPUTS["summary"]
    assert result["key_findings"] == FAKE_RESULT_OUTPUTS["key_findings"]
    assert result["sql_hash"] == "deadbeefdeadbeef"
    assert result["total_tokens"] == 321
    assert result["nodes"][0]["node_id"] == "analyst_1"
    # 按调用点的成本归因要透出到接口层（分级路由与成本计量的依据）
    assert result["tokens_by_route"]["planner"]["calls"] == 1


def test_pipeline_crash_is_reported_not_propagated() -> None:
    """任务崩溃必须转成 failed 状态，不能让服务端 500。"""
    client = _client(should_fail=True)
    task_id = client.post("/tasks", json={"question": "q"}).json()["task_id"]
    body = _wait_finished(client, task_id)
    assert body["status"] == "failed"
    assert "模拟流水线崩溃" in body["error"]


def test_unknown_task_returns_404() -> None:
    client = _client()
    assert client.get("/tasks/does-not-exist").status_code == 404
    assert client.get("/tasks/does-not-exist/events").status_code == 404


def test_task_list_is_returned() -> None:
    client = _client()
    client.post("/tasks", json={"question": "第一个问题"})
    body = client.get("/tasks").json()
    assert body["tasks"], "应有至少一条任务"
    assert {"task_id", "question", "status", "created_at"} <= set(body["tasks"][0])


# ------------------------------------------------------------------ SSE 事件


def test_events_stream_contains_node_progress_and_finish() -> None:
    client = _client()
    task_id = client.post("/tasks", json={"question": "q"}).json()["task_id"]
    _wait_finished(client, task_id)

    with client.stream("GET", f"/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "".join(response.iter_text())

    assert "event: task_started" in payload
    assert "event: node_started" in payload
    assert "event: node_finished" in payload
    assert "event: task_finished" in payload
    # 事件必须带 node_id 与 agent_role，否则前端无法把进度挂到节点上
    assert "planner_1" in payload and "analyst_1" in payload
    assert "retriever_2" in payload, "并行检索的第二路也应出现在事件流里"
    assert '"agent_role": "retriever"' in payload


def test_streaming_after_completion_replays_buffered_events() -> None:
    """晚连上来的客户端也应看到完整过程，而不是空流。"""
    client = _client()
    task_id = client.post("/tasks", json={"question": "q"}).json()["task_id"]
    _wait_finished(client, task_id)

    with client.stream("GET", f"/tasks/{task_id}/events") as response:
        payload = "".join(response.iter_text())

    # planner（图外）+ 两路并行检索 + 分析
    assert payload.count("event: node_finished") == 4
    assert "task_finished" in payload


def test_events_and_result_expose_reviewer_and_blackboard() -> None:
    """W6：SSE 里能看到 reviewer 节点与黑板变更事件；接口结果里带评审结论。

    这条是「评审链路对观测面可见」的确定性测试——不依赖真实模型与数据库。
    """
    app = create_app(
        settings=Settings(_env_file=None),
        pipeline_factory=lambda s, on_event: FakePipeline(s, on_event, with_review=True),
    )
    client = TestClient(app)
    task_id = client.post("/tasks", json={"question": "q"}).json()["task_id"]
    body = _wait_finished(client, task_id)

    assert body["status"] == "ok"
    # 结果里暴露评审结论（能不能采信是 verdict 说了算）
    assert body["result"]["review"]["verdict"] == "pass"
    assert body["result"]["review"]["issue_count"] == 0

    with client.stream("GET", f"/tasks/{task_id}/events") as response:
        payload = "".join(response.iter_text())

    assert '"agent_role": "reviewer"' in payload
    assert "event: blackboard_written" in payload
    # 黑板事件要带 key 与版本，订阅方才能据此判断「哪条、第几版」
    assert '"key": "analyst_1.report"' in payload
    assert '"version": 2' in payload and '"status": "validated"' in payload


def test_every_event_is_valid_sse_with_json_data() -> None:
    import json

    client = _client()
    task_id = client.post("/tasks", json={"question": "中文问题 ✓"}).json()["task_id"]
    _wait_finished(client, task_id)

    with client.stream("GET", f"/tasks/{task_id}/events") as response:
        raw = "".join(response.iter_text())

    blocks = [b for b in raw.split("\n\n") if b.strip()]
    assert blocks
    for block in blocks:
        lines = block.splitlines()
        assert lines[0].startswith("event: "), block
        data_line = next((ln for ln in lines if ln.startswith("data: ")), None)
        assert data_line, block
        json.loads(data_line.removeprefix("data: "))  # 必须是合法 JSON
