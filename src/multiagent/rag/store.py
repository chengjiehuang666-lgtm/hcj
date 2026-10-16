"""知识库存取：分块 + 向量入 pgvector + 混合检索。

**写入与检索用不同的数据库角色**（与 SQL 沙箱的权限设计一致）：
- 灌库（ingest）用应用主账号，属管理操作；
- 检索（search）用只读角色 `analyst_ro`，与 Agent 的其它查询同权限。

**混合检索**：向量分与关键词分加权融合。
- 向量分：`1 - 余弦距离`，负责语义相近（「卖得差」↔「销售额下滑」）；
- 关键词分：`pg_trgm similarity()`，负责专有名词与数字的精确命中
  （「退货政策」「37.6%」这类向量未必抓得住的东西）。

不用 RRF（倒数排名融合）而用线性加权，是因为两路分数都已归一化到 [0,1]，
线性加权更直观、便于按语料调权重；RRF 的优势在排名尺度不可比时才体现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..config import Settings
from .chunking import Chunk, chunk_document
from .embeddings import LocalEmbedder, get_shared_embedder, to_pgvector_literal


class KnowledgeStoreError(Exception):
    """知识库读写失败。"""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_index: int
    title: str
    content: str
    vector_score: float
    keyword_score: float
    score: float

    def as_prompt_text(self) -> str:
        return f"《{self.title}》片段 {self.chunk_index}（相关度 {self.score:.2f}）：{self.content}"


class KnowledgeStore:
    def __init__(self, settings: Settings, embedder: LocalEmbedder | None = None) -> None:
        self._settings = settings
        # 默认复用进程内共享的 embedder，避免每个任务都重新加载模型
        self._embedder = embedder or get_shared_embedder(settings)

    # ------------------------------------------------------------------ 写入

    def ingest(self, documents: Iterable[Document]) -> dict[str, int]:
        """分块、向量化并写入。同 (doc_id, chunk_index) 重复灌库为覆盖更新。"""
        import psycopg

        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(
                chunk_document(
                    doc.doc_id,
                    doc.title,
                    doc.text,
                    chunk_size=self._settings.rag_chunk_size,
                    overlap=self._settings.rag_chunk_overlap,
                )
            )
        if not chunks:
            return {"documents": 0, "chunks": 0}

        vectors = self._embedder.embed([c.content for c in chunks])
        rows = [
            (
                c.doc_id,
                c.chunk_index,
                c.title,
                c.content,
                to_pgvector_literal(vec),
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]

        sql = """
            INSERT INTO knowledge_chunks (doc_id, chunk_index, title, content, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (doc_id, chunk_index) DO UPDATE
               SET title = EXCLUDED.title,
                   content = EXCLUDED.content,
                   embedding = EXCLUDED.embedding,
                   created_at = now()
        """
        try:
            with psycopg.connect(self._settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.executemany(sql, rows)
        except Exception as exc:
            raise KnowledgeStoreError(f"写入知识库失败: {exc}") from exc

        doc_count = len({c.doc_id for c in chunks})
        return {"documents": doc_count, "chunks": len(chunks)}

    # ------------------------------------------------------------------ 检索

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        import psycopg
        from psycopg.rows import dict_row

        if not query.strip():
            return []
        k = top_k or self._settings.rag_top_k
        weight = self._settings.rag_vector_weight
        vector_literal = to_pgvector_literal(self._embedder.embed_one(query))

        sql = """
            SELECT doc_id, chunk_index, title, content,
                   1 - (embedding <=> %(vec)s::vector)      AS vector_score,
                   COALESCE(similarity(content, %(q)s), 0)  AS keyword_score,
                   %(vw)s * (1 - (embedding <=> %(vec)s::vector))
                     + %(kw)s * COALESCE(similarity(content, %(q)s), 0) AS score
              FROM knowledge_chunks
             WHERE embedding IS NOT NULL
             ORDER BY score DESC NULLS LAST
             LIMIT %(k)s
        """
        params = {
            "vec": vector_literal,
            "q": query,
            "vw": weight,
            "kw": 1.0 - weight,
            "k": k,
        }
        try:
            with psycopg.connect(
                self._settings.postgres_ro_dsn, connect_timeout=10, row_factory=dict_row
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        except Exception as exc:
            raise KnowledgeStoreError(f"检索知识库失败: {exc}") from exc

        return [
            RetrievedChunk(
                doc_id=r["doc_id"],
                chunk_index=r["chunk_index"],
                title=r["title"],
                content=r["content"],
                vector_score=float(r["vector_score"]),
                keyword_score=float(r["keyword_score"]),
                score=float(r["score"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ 统计

    def stats(self) -> dict[str, Any]:
        import psycopg

        sql = (
            "SELECT count(*) AS chunks, count(DISTINCT doc_id) AS documents, "
            "count(embedding) AS embedded FROM knowledge_chunks"
        )
        try:
            with psycopg.connect(self._settings.postgres_ro_dsn, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
        except Exception as exc:
            raise KnowledgeStoreError(f"读取知识库统计失败: {exc}") from exc
        return {"chunks": row[0], "documents": row[1], "embedded": row[2]}
