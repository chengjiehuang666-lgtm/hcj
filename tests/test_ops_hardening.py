"""W14 安全/运维加固：审计日志、限流、安全响应头。"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from multiagent.api import create_app  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.graph.pipeline import PipelineOutcome  # noqa: E402

KEYS = "admin-key:admin,op-key:operator,view-key:viewer"


class FakePipeline:
    def __init__(self, settings, on_event=None) -> None:
        pass

    async def run(self, event, *, resume=False, resume_value=None):
        return PipelineOutcome(
            task_id=str(event.task_id),
            status="ok",
            summary="ok",
            total_tokens=10,
            cost=0.1,
            duration_ms=5,
            node_results=[],
        )


def _settings(tmp_path: pathlib.Path, **overrides) -> Settings:
    base = dict(
        _env_file=None,
        audit_enabled=True,
        audit_log_path=str(tmp_path / "audit.jsonl"),
        rate_limit_enabled=False,
    )
    base.update(overrides)
    return Settings(**base)


def _audit_lines(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _wait(client: TestClient, task_id: str, key: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    for _ in range(50):
        body = client.get(f"/tasks/{task_id}", headers=headers).json()
        if body["status"] in ("ok", "failed", "rejected"):
            return body
    raise AssertionError("任务未结束")


def test_audit_records_submit_and_finish_without_raw_question(tmp_path: pathlib.Path) -> None:
    settings = _settings(tmp_path, api_auth_enabled=True, api_keys=KEYS)
    app = create_app(settings=settings, pipeline_factory=FakePipeline)
    client = TestClient(app)
    question = "分析手机号 13812345678 的订单"
    response = client.post(
        "/tasks", json={"question": question}, headers={"Authorization": "Bearer op-key"}
    )
    assert response.status_code == 202
    _wait(client, response.json()["task_id"], "view-key")

    lines = _audit_lines(tmp_path / "audit.jsonl")
    events = [item["event"] for item in lines]
    assert "task_submit" in events and "task_finished" in events
    submit = next(item for item in lines if item["event"] == "task_submit")
    assert submit["actor"] and submit["question_len"] > 0
    assert len(submit["question_sha256"]) == 12
    # 原始问题（含手机号）不能出现在审计里
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "13812345678" not in raw


def test_audit_records_auth_failure(tmp_path: pathlib.Path) -> None:
    settings = _settings(tmp_path, api_auth_enabled=True, api_keys=KEYS)
    client = TestClient(create_app(settings=settings, pipeline_factory=FakePipeline))
    assert client.get("/tasks").status_code == 401
    lines = _audit_lines(tmp_path / "audit.jsonl")
    assert any(item["event"] == "auth_failed" and item["reason"] == "missing_key" for item in lines)


def test_rate_limit_returns_429_and_health_is_exempt(tmp_path: pathlib.Path) -> None:
    settings = _settings(
        tmp_path,
        rate_limit_enabled=True,
        rate_limit_per_minute=2,
        rate_limit_burst=2,
    )
    client = TestClient(create_app(settings=settings, pipeline_factory=FakePipeline))
    assert client.get("/tasks").status_code == 200
    assert client.get("/tasks").status_code == 200
    blocked = client.get("/tasks")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert client.get("/health").status_code in (200, 503)
    assert any(item["event"] == "rate_limited" for item in _audit_lines(tmp_path / "audit.jsonl"))


def test_rate_limit_is_per_key(tmp_path: pathlib.Path) -> None:
    settings = _settings(
        tmp_path,
        api_auth_enabled=True,
        api_keys=KEYS,
        rate_limit_enabled=True,
        rate_limit_per_minute=1,
        rate_limit_burst=1,
    )
    client = TestClient(create_app(settings=settings, pipeline_factory=FakePipeline))
    assert client.get("/tasks", headers={"Authorization": "Bearer op-key"}).status_code == 200
    assert client.get("/tasks", headers={"Authorization": "Bearer op-key"}).status_code == 429
    # 另一个 Key 有自己的桶
    assert client.get("/tasks", headers={"Authorization": "Bearer view-key"}).status_code == 200


def test_security_headers_and_hsts_on_https(tmp_path: pathlib.Path) -> None:
    settings = _settings(tmp_path)
    client = TestClient(
        create_app(settings=settings, pipeline_factory=FakePipeline),
        base_url="https://testserver",
    )
    response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Strict-Transport-Security"].startswith("max-age=")
