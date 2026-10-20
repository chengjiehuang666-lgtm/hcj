"""长期记忆的存取与沉淀规则。

设计要点：

- **沉淀有门槛**（`distill_memory`）：任务成功 + 评审未否决 + 结论明确存在且不是
  「数据不足」。门槛写成纯函数并单测钉住——它是「错误记忆永久污染」的第一道闸门。
- **读写分角色**：写用应用主账号（管理操作），检索用只读角色（analyst_ro），
  与知识库、SQL 沙箱同一套权限口径。
- **检索只取 `status = 'validated'`**：被评审否决或已过期的条目对后续会话不可见。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..rag.embeddings import LocalEmbedder, get_shared_embedder, to_pgvector_literal
from ..security.dlp import redact_entry


class MemoryStoreError(Exception):
    """长期记忆读写失败。"""


@dataclass
class MemoryHit:
    memory_id: int
    kind: str
    content: str
    source_ref: str
    confidence: float | None
    created_at: str
    vector_score: float
    keyword_score: float
    score: float

    def as_prompt_text(self) -> str:
        return (
            f"（历史结论，来源：{self.source_ref}，相关度 {self.score:.2f}）"
            f"{self.content}"
        )


def distill_memory(outcome) -> dict[str, Any] | None:
    """从一次任务产出提炼可沉淀的记忆条目；返回 None 表示不沉淀。

    门槛（全部满足才沉淀，缺一不可）：
    1. 任务成功；
    2. **存在评审节点且没有任何 reviewer 给出 fail**——「被标记为可沉淀」的落点；
       没有评审 = 没被标记，同样不沉淀；
    3. analyst 结论存在、非空、且不是「数据不足」（数据不足的结论对后续会话
       价值低，且有被误当作事实引用的风险）；
    4. 清洗：只保留结论与关键发现（不含原始行数据、不含 SQL 全文），
       来源写入 source_ref（任务 + SQL 哈希）供逐条回溯。

    这是纯函数（只读 outcome，不碰数据库），因此门槛本身可以被单测穷举。
    """
    if getattr(outcome, "status", None) != "ok":
        return None

    reviewers = [
        r for r in outcome.node_results if r.get("agent_role") == "reviewer"
    ]
    if not reviewers:
        return None
    if any(
        (r.get("outputs") or {}).get("verdict") == "fail" for r in reviewers
    ):
        return None

    analysts = [
        r
        for r in outcome.node_results
        if r.get("agent_role") == "analyst" and r.get("status") == "ok"
    ]
    if not analysts:
        return None
    outputs: dict[str, Any] = {}
    for r in analysts:
        outputs.update(r.get("outputs") or {})
    if outputs.get("insufficient_data"):
        return None

    summary = str(outputs.get("summary") or "").strip()
    if not summary:
        return None
    findings = [
        str(f).strip() for f in (outputs.get("key_findings") or []) if str(f).strip()
    ]
    content = summary
    if findings:
        content += "\n关键发现：\n" + "\n".join(f"- {f}" for f in findings)
    content = content[:1200]

    last = analysts[-1]
    sql_hash = outputs.get("sql_hash")
    source_ref = f"任务 {outcome.task_id}" + (
        f"，SQL sha256:{sql_hash}" if sql_hash else ""
    )
    return {
        "task_id": str(outcome.task_id),
        "node_id": str(last.get("node_id") or "analyst_1"),
        "kind": "conclusion",
        "content": content,
        "source_type": "model_inference",
        "source_ref": source_ref,
        "confidence": last.get("confidence"),
    }


class MemoryStore:
    def __init__(self, settings: Settings, embedder: LocalEmbedder | None = None) -> None:
        self._settings = settings
        # 与知识库共用进程内 embedder（同一个模型、同一份缓存，避免重复加载）
        self._embedder = embedder or get_shared_embedder(settings)

    # ------------------------------------------------------------------ 写入

    def sink(self, entry: dict[str, Any]) -> int | None:
        """写入一条记忆；内容已存在时**刷新其来源**。

        **判重必须按内容**：每次任务运行都是新 task_id（UUID），仅靠
        `UNIQUE (task_id)` 的去重对「同题重跑 / 多轮评测」永不生效——独立复核实测
        一轮 27 条评测就给库里堆了 19 条近重复结论。

        命中同内容时不新增行，而是把 source_ref 刷新为本次运行——「记忆来自本次
        运行」的绑定断言（eval 的记忆用例）依赖它：刷新后回归里的 source_ref
        携带本次前置用例的 task_id，既去了重又保住了牙齿。返回条目 id
        （新增或命中都返回；仅同任务重复写入时为 None）。
        """
        import psycopg

        vector = to_pgvector_literal(self._embedder.embed_one(entry["content"]))
        find_sql = "SELECT id FROM memory_entries WHERE content = %s LIMIT 1"
        refresh_sql = "UPDATE memory_entries SET source_ref = %s WHERE id = %s"
        insert_sql = """
            INSERT INTO memory_entries
                   (task_id, node_id, kind, content, embedding,
                    source_type, source_ref, confidence, status)
            VALUES (%(task_id)s, %(node_id)s, %(kind)s, %(content)s, %(vec)s::vector,
                    %(source_type)s, %(source_ref)s, %(confidence)s, 'validated')
            ON CONFLICT (task_id) DO NOTHING
            RETURNING id
        """
        params = {**entry, "vec": vector}
        try:
            with psycopg.connect(self._settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(find_sql, (entry["content"],))
                    existing = cur.fetchone()
                    if existing is not None:
                        cur.execute(refresh_sql, (entry["source_ref"], existing[0]))
                        return int(existing[0])
                    cur.execute(insert_sql, params)
                    row = cur.fetchone()
        except Exception as exc:
            raise MemoryStoreError(f"写入长期记忆失败: {exc}") from exc
        return int(row[0]) if row else None

    def sink_outcome(self, outcome) -> int | None:
        """从任务产出沉淀（门槛见 `distill_memory`）；不可沉淀时返回 None。"""
        entry = distill_memory(outcome)
        if entry is None:
            return None
        # 输出侧 DLP：记忆是长期存档，绝不能把身份证/手机号/密钥写进库
        if self._settings.security_dlp_enabled:
            entry = redact_entry(entry)
        return self.sink(entry)

    # ------------------------------------------------------------------ 检索

    def search(self, query: str, *, top_k: int | None = None) -> list[MemoryHit]:
        """混合检索（与知识库同一套加权公式）；只返回 validated 且未过期的条目。"""
        import psycopg
        from psycopg.rows import dict_row

        if not query.strip():
            return []
        k = top_k or self._settings.memory_top_k
        weight = self._settings.rag_vector_weight
        vector_literal = to_pgvector_literal(self._embedder.embed_one(query))

        sql = """
            SELECT id, kind, content, source_ref, confidence, created_at,
                   1 - (embedding <=> %(vec)s::vector)      AS vector_score,
                   COALESCE(similarity(content, %(q)s), 0)  AS keyword_score,
                   %(vw)s * (1 - (embedding <=> %(vec)s::vector))
                     + %(kw)s * COALESCE(similarity(content, %(q)s), 0) AS score
              FROM memory_entries
             WHERE status = 'validated'
               AND embedding IS NOT NULL
               AND (ttl_seconds IS NULL
                    OR created_at + make_interval(secs => ttl_seconds) > now())
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
            raise MemoryStoreError(f"检索长期记忆失败: {exc}") from exc

        return [
            MemoryHit(
                memory_id=int(r["id"]),
                kind=r["kind"],
                content=r["content"],
                source_ref=r["source_ref"],
                confidence=r["confidence"],
                created_at=str(r["created_at"]),
                vector_score=float(r["vector_score"]),
                keyword_score=float(r["keyword_score"]),
                score=float(r["score"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ 管理

    def stats(self) -> dict[str, int]:
        import psycopg

        sql = (
            "SELECT count(*) AS total, "
            "count(*) FILTER (WHERE status = 'validated') AS validated "
            "FROM memory_entries"
        )
        try:
            with psycopg.connect(self._settings.postgres_ro_dsn, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
        except Exception as exc:
            raise MemoryStoreError(f"读取长期记忆统计失败: {exc}") from exc
        return {"total": int(row[0]), "validated": int(row[1])}

    def clear(self) -> int:
        """清空记忆库（运维/测试操作，走应用主账号）。返回删除条数。"""
        import psycopg

        try:
            with psycopg.connect(self._settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM memory_entries")
                    deleted = cur.rowcount
        except Exception as exc:
            raise MemoryStoreError(f"清空长期记忆失败: {exc}") from exc
        return int(deleted)
