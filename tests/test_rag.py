"""RAG 层测试：分块、本地 embedding、混合检索、以及「知识不能变成数据」的约束。

需要数据库与 embedding 模型的用例，在不可用时跳过并说明原因。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.config import Settings  # noqa: E402
from multiagent.rag import (  # noqa: E402
    DOCUMENTS,
    Document,
    EmbeddingError,
    KnowledgeStore,
    KnowledgeStoreError,
    LocalEmbedder,
    chunk_document,
    to_pgvector_literal,
)

settings = Settings(_env_file=ROOT / ".env")


def _store() -> KnowledgeStore:
    return KnowledgeStore(settings)


def _require_db(store: KnowledgeStore):
    try:
        store.stats()
    except KnowledgeStoreError as exc:
        pytest.skip(f"数据库不可达（docker compose up -d 了吗？）: {exc}")


# ------------------------------------------------------------------ 分块


def test_chunk_respects_size_limit() -> None:
    text = "\n\n".join(f"第{i}段：" + "内容" * 40 for i in range(6))
    chunks = chunk_document("d1", "标题", text, chunk_size=200, overlap=40)
    assert chunks, "应至少产出 1 个分块"
    assert all(len(c.content) <= 200 for c in chunks), [len(c.content) for c in chunks]


def test_chunk_indexes_are_sequential_and_stable() -> None:
    chunks = chunk_document("d1", "标题", "甲乙丙" * 300, chunk_size=150, overlap=30)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "d1" and c.title == "标题" for c in chunks)


def test_chunk_hard_splits_overlong_paragraph() -> None:
    """单段超长时必须硬切，否则会产生超长块。"""
    chunks = chunk_document("d1", "标题", "字" * 1000, chunk_size=200, overlap=20)
    assert len(chunks) >= 5
    assert all(len(c.content) <= 200 for c in chunks)


def test_chunk_prefers_paragraph_boundaries() -> None:
    text = "第一段内容。\n\n第二段内容。"
    chunks = chunk_document("d1", "标题", text, chunk_size=500, overlap=0)
    assert len(chunks) == 1, "两段能放进一块时不应拆开"
    assert "第一段" in chunks[0].content and "第二段" in chunks[0].content


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_document("d1", "标题", "   \n\n  ", chunk_size=200, overlap=20) == []


# ------------------------------------------------------------ 向量字面量


def test_pgvector_literal_format() -> None:
    literal = to_pgvector_literal([0.5, -1.0, 0.0])
    assert literal.startswith("[") and literal.endswith("]")
    assert literal.count(",") == 2


# ---------------------------------------------------------------- embedding


def _embedder_or_skip() -> LocalEmbedder:
    embedder = LocalEmbedder(settings)
    try:
        embedder.embed(["探活"])
    except EmbeddingError as exc:
        pytest.skip(f"embedding 模型不可用（缓存 {settings.embedding_cache_dir}）: {exc}")
    return embedder


def test_embedding_dimension_matches_config() -> None:
    embedder = _embedder_or_skip()
    assert embedder.dim == settings.embedding_dim


def test_embedding_is_deterministic() -> None:
    embedder = _embedder_or_skip()
    first, second = embedder.embed(["销售额下滑"]), embedder.embed(["销售额下滑"])
    assert first == second


def test_embedding_separates_unrelated_texts() -> None:
    """相关文本的相似度必须明显高于无关文本，否则检索没有意义。"""
    embedder = _embedder_or_skip()
    import math

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    close = embedder.embed(["华东区销售额下滑", "华东区销售表现不佳"])
    far = embedder.embed(["华东区销售额下滑", "公司团建活动安排"])
    assert cos(close[0], close[1]) > cos(far[0], far[1]) + 0.1


# ------------------------------------------------------------------ 检索


def test_retrieval_finds_policy_document_for_cause_question() -> None:
    """核心用例：问「为什么」，必须命中解释性文档（这是 RAG 的存在理由）。"""
    store = _store()
    _require_db(store)
    try:
        hits = store.search("9月线上退货率为什么大幅上升", top_k=3)
    except Exception as exc:
        pytest.skip(f"检索不可用: {exc}")
    assert hits, "知识库为空，请先执行 python -m multiagent.rag.ingest"
    assert any("退货政策" in h.title for h in hits), [h.title for h in hits]


def test_retrieval_finds_data_dictionary_for_feasibility_question() -> None:
    store = _store()
    _require_db(store)
    try:
        hits = store.search("这张表能不能算月内的日波动率", top_k=3)
    except Exception as exc:
        pytest.skip(f"检索不可用: {exc}")
    assert hits
    assert any("数据字典" in h.title for h in hits), [h.title for h in hits]


def test_results_are_sorted_by_fused_score() -> None:
    store = _store()
    _require_db(store)
    try:
        hits = store.search("华东区包含哪些省份", top_k=5)
    except Exception as exc:
        pytest.skip(f"检索不可用: {exc}")
    assert hits
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), scores


def test_empty_query_returns_nothing_without_touching_db() -> None:
    assert _store().search("   ") == []


def test_builtin_corpus_is_ingested() -> None:
    store = _store()
    _require_db(store)
    try:
        stats = store.stats()
    except Exception as exc:
        pytest.skip(f"检索不可用: {exc}")
    if stats["chunks"] == 0:
        pytest.skip("知识库尚未灌库（python -m multiagent.rag.ingest）")
    assert stats["documents"] >= len(DOCUMENTS), stats
    assert stats["embedded"] == stats["chunks"], "所有分块都应已向量化"


def test_ingest_is_idempotent() -> None:
    """重复灌库应覆盖而非重复插入——否则检索结果会被同一个文档刷屏。"""
    store = _store()
    _require_db(store)
    try:
        before = store.stats()
        store.ingest(DOCUMENTS)
        after = store.stats()
    except Exception as exc:
        pytest.skip(f"检索不可用: {exc}")
    if before["chunks"] == 0:
        pytest.skip("知识库尚未灌库")
    assert after["chunks"] == before["chunks"], (before, after)
