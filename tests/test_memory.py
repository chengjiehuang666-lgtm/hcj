"""长期记忆测试。

两层：
1. **沉淀门槛**是纯函数（`distill_memory`），穷举各条否决条件——它是「错误记忆
   永久污染」的第一道闸门，必须逐条钉住；
2. **存取**（sink/search/stats/clear）需要数据库，不可达则跳过；embedder 注入
   假实现，避免测试加载真实模型。
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.config import Settings  # noqa: E402
from multiagent.memory import MemoryStore, MemoryStoreError, distill_memory  # noqa: E402

settings = Settings(_env_file=ROOT / ".env")


class FakeEmbedder:
    """恒定向量：让关键词分成为唯一区分项，检索行为可预测且不加载模型。"""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def dim(self) -> int:
        return 512

    def embed_one(self, text: str) -> list[float]:
        self.calls += 1
        return [0.01] * 512

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def _node(
    role: str,
    *,
    outputs: dict | None = None,
    status: str = "ok",
    confidence: float | None = 0.8,
) -> dict:
    return {
        "node_id": f"{role}_1",
        "agent_role": role,
        "status": status,
        "confidence": confidence,
        "outputs": outputs if outputs is not None else {},
    }


def _outcome(
    *,
    status: str = "ok",
    nodes: list[dict] | None = None,
    task_id: str = "t-1",
) -> SimpleNamespace:
    return SimpleNamespace(status=status, task_id=task_id, node_results=nodes or [])


_ANALYST = _node(
    "analyst",
    outputs={
        "summary": "华东区 9 月销售额环比下降 20.85%",
        "key_findings": ["线上数码降 37.6%", "门店仅降 1.95%"],
        "sql_hash": "abc123",
        "row_count": 8,
        "rows_preview": [{"region": "华东", "revenue": 2335000}],
        "insufficient_data": False,
    },
)
_REVIEW_PASS = _node("reviewer", outputs={"verdict": "pass", "issue_count": 0})


# ---------------------------------------------------------------- 沉淀门槛


def test_distill_happy_path() -> None:
    entry = distill_memory(_outcome(nodes=[_ANALYST, _REVIEW_PASS]))

    assert entry is not None
    assert entry["kind"] == "conclusion"
    assert entry["task_id"] == "t-1"
    assert "环比下降 20.85%" in entry["content"]
    assert "线上数码降 37.6%" in entry["content"]
    # 清洗：不得把原始行数据或 SQL 本体带进记忆
    assert "rows_preview" not in entry["content"] and "SELECT" not in entry["content"]
    # 来源可回溯：任务 + SQL 哈希
    assert "t-1" in entry["source_ref"] and "abc123" in entry["source_ref"]
    assert entry["source_type"] == "model_inference"


def test_distill_requires_successful_task() -> None:
    assert distill_memory(_outcome(status="failed", nodes=[_ANALYST, _REVIEW_PASS])) is None


def test_distill_requires_reviewer() -> None:
    """没有评审 = 没被标记为可沉淀：宁可漏沉淀，不可未审先沉淀。"""
    assert distill_memory(_outcome(nodes=[_ANALYST])) is None


def test_distill_rejects_reviewer_fail() -> None:
    reviewer = _node("reviewer", outputs={"verdict": "fail", "issue_count": 1})
    assert distill_memory(_outcome(nodes=[_ANALYST, reviewer])) is None


def test_distill_allows_pass_with_issues() -> None:
    reviewer = _node("reviewer", outputs={"verdict": "pass_with_issues", "issue_count": 2})
    assert distill_memory(_outcome(nodes=[_ANALYST, reviewer])) is not None


def test_distill_rejects_insufficient_data() -> None:
    analyst = _node(
        "analyst",
        outputs={
            "summary": "数据不足，无法判断",
            "key_findings": [],
            "insufficient_data": True,
        },
    )
    assert distill_memory(_outcome(nodes=[analyst, _REVIEW_PASS])) is None


def test_distill_truncates_content() -> None:
    analyst = _node(
        "analyst",
        outputs={"summary": "长" * 2000, "key_findings": [], "insufficient_data": False},
    )
    entry = distill_memory(_outcome(nodes=[analyst, _REVIEW_PASS]))
    assert entry is not None and len(entry["content"]) <= 1200


# ---------------------------------------------------------------- 存取（需数据库）


def _store() -> MemoryStore:
    return MemoryStore(settings, embedder=FakeEmbedder())


def _db_ready() -> None:
    import psycopg

    try:
        with psycopg.connect(settings.postgres_ro_dsn, connect_timeout=5):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可达: {exc}")
    try:
        _store().stats()
    except MemoryStoreError as exc:
        pytest.skip(f"记忆表不可用: {exc}")


def test_sink_search_dedup_and_clear() -> None:
    _db_ready()
    store = _store()
    store.clear()

    entry = distill_memory(_outcome(nodes=[_ANALYST, _REVIEW_PASS]))
    assert entry is not None
    memory_id = store.sink(entry)
    assert memory_id is not None

    # 重复沉淀（同 task_id）按已有条目处理，不产生第二条
    assert store.sink(entry) is None
    stats = store.stats()
    assert stats == {"total": 1, "validated": 1}

    hits = store.search("华东区销售额环比下降")
    assert hits and hits[0].memory_id == memory_id
    assert "20.85%" in hits[0].content
    assert "t-1" in hits[0].source_ref

    assert store.clear() == 1
    assert store.search("华东区销售额环比下降") == []


def test_search_only_returns_validated() -> None:
    """draft/rejected 条目对后续会话不可见——评审未通过的东西不得回流。"""
    _db_ready()
    import psycopg

    store = _store()
    store.clear()
    entry = distill_memory(_outcome(nodes=[_ANALYST, _REVIEW_PASS]))
    assert entry is not None
    store.sink(entry)

    # 直接插入一条 draft（模拟未经评审的条目）
    with psycopg.connect(settings.postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory_entries (task_id, node_id, kind, content, "
                "source_type, source_ref, status) VALUES "
                "('t-draft', 'analyst_1', 'conclusion', '华东区销售额环比下降草稿', "
                "'model_inference', '任务 t-draft', 'draft')"
            )

    hits = store.search("华东区销售额环比下降")
    assert len(hits) == 1 and hits[0].memory_id != 0
    assert all("草稿" not in h.content for h in hits)
    store.clear()


def test_retriever_memory_mode(monkeypatch) -> None:
    """retriever 的 memory 模式：命中进 outputs.memories，未命中留来源记录。"""
    _db_ready()
    from multiagent.agents import RetrieverAgent
    from multiagent.agents.base import AgentContext
    from multiagent.contracts import AgentInvocation, InvocationBudget
    from multiagent.tools import SqlSandbox

    store = _store()
    store.clear()
    entry = distill_memory(_outcome(nodes=[_ANALYST, _REVIEW_PASS]))
    assert entry is not None
    store.sink(entry)

    # Retriever 内部自建 MemoryStore：注入假 embedder 避免加载真模型
    import multiagent.agents.retriever as retriever_module

    monkeypatch.setattr(
        retriever_module, "MemoryStore", lambda s: MemoryStore(s, embedder=FakeEmbedder())
    )

    inv = AgentInvocation(
        task_id="t-1",
        node_id="retriever_3",
        attempt=1,
        agent_role="retriever",
        trace_id="t-1",
        input={"question": "上次分析华东区下滑的结论是什么", "mode": "memory"},
        budget=InvocationBudget(max_tokens=1000, timeout_seconds=60),
    )
    ctx = AgentContext(
        settings=settings,
        gateway=None,  # memory 模式不调用模型
        sandbox=SqlSandbox(settings),
    )
    result = asyncio.run(RetrieverAgent().invoke(inv, ctx))

    assert result.status.value == "ok"
    assert result.outputs["memory_count"] == 1
    assert "20.85%" in result.outputs["memories"][0]["content"]
    assert any("长期记忆:" in s.ref for s in result.sources)
    assert result.blackboard_writes == ["retriever_3.memory"]

    store.clear()
    result = asyncio.run(RetrieverAgent().invoke(inv, ctx))
    assert result.outputs["memory_count"] == 0
    # 未命中也是结果：必须有来源记录（契约要求 status=ok 时 sources 非空）
    assert any("无命中" in s.ref for s in result.sources)


# ---------------------------------------------------------------- 沉淀钩子（best-effort）


def test_pipeline_sink_hook_is_best_effort(monkeypatch) -> None:
    """沉淀失败不得影响任务：错误只记进 outcome.memory_error。"""
    from multiagent.graph import pipeline as pipeline_module
    from multiagent.graph.pipeline import AnalysisPipeline, PipelineOutcome

    events: list[tuple[str, dict]] = []

    async def on_event(name, payload):
        events.append((name, payload))

    class BrokenStore:
        def __init__(self, settings):
            pass

        def sink_outcome(self, outcome):
            raise MemoryStoreError("库里没有这张表")

    monkeypatch.setattr(pipeline_module, "MemoryStore", BrokenStore)
    pipe = AnalysisPipeline(settings, on_event=on_event)
    outcome = PipelineOutcome(task_id="t-1", status="ok")

    asyncio.run(pipe._sink_memory(outcome))  # 不抛异常

    assert outcome.memory_sunk is None
    assert "库里没有这张表" in (outcome.memory_error or "")
    assert any(name == "memory_sink_failed" for name, _ in events)


def test_pipeline_sink_hook_records_success(monkeypatch) -> None:
    from multiagent.graph import pipeline as pipeline_module
    from multiagent.graph.pipeline import AnalysisPipeline, PipelineOutcome

    events: list[tuple[str, dict]] = []

    async def on_event(name, payload):
        events.append((name, payload))

    class OkStore:
        def __init__(self, settings):
            pass

        def sink_outcome(self, outcome):
            return 42

    monkeypatch.setattr(pipeline_module, "MemoryStore", OkStore)
    pipe = AnalysisPipeline(settings, on_event=on_event)
    outcome = PipelineOutcome(task_id="t-1", status="ok")

    asyncio.run(pipe._sink_memory(outcome))

    assert outcome.memory_sunk == 42
    assert any(name == "memory_sunk" for name, _ in events)
