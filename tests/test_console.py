"""W13 轻量控制台：静态页面 + 结果里的 DAG 数据。"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from multiagent.api import create_app  # noqa: E402
from multiagent.api.app import _outcome_to_result  # noqa: E402
from multiagent.config import Settings  # noqa: E402
from multiagent.graph.pipeline import PipelineOutcome  # noqa: E402


def _client() -> TestClient:
    return TestClient(create_app(settings=Settings(_env_file=None)))


def test_console_page_and_assets_are_served() -> None:
    client = _client()
    page = client.get("/console/")
    assert page.status_code == 200
    assert "多 Agent 控制台" in page.text
    assert "console.js" in page.text
    assert client.get("/console/console.js").status_code == 200
    assert client.get("/console/console.css").status_code == 200
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307)
    assert root.headers["location"] == "/console/"


def test_outcome_result_includes_dag_for_visualization() -> None:
    outcome = PipelineOutcome(
        task_id="t1",
        status="ok",
        dag={
            "nodes": [
                {"node_id": "retriever_1", "agent_role": "retriever", "depends_on": []},
                {"node_id": "analyst_1", "agent_role": "analyst", "depends_on": ["retriever_1"]},
            ]
        },
        node_results=[],
    )
    result = _outcome_to_result(outcome)
    assert result["dag"]["nodes"][1]["depends_on"] == ["retriever_1"]
