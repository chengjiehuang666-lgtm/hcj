"""知识检索 Agent（Retriever）。

职责：为后续分析准备两类上下文。
1. **结构化口径**：直接查 `metric_definitions`（SQL 检索）；
2. **非结构化知识**：从知识库做混合检索（向量 + 关键词），拿到 SQL 查不出的
   业务背景（政策变更、历史复盘、数据口径注意事项）。

设计取舍：
- 口径数据刻意**不让模型转述**——结构化数据直查更准、更省 token，
  也避免模型在「口径」这种不允许失真的信息上引入幻觉；
- 知识库检索同理不经过模型：embedding + SQL 加权排序即可，交给模型转述只会失真；
- **RAG 失败不让整个任务失败**：知识是增强而非必需，检索不到就降级，
  但在结果里显式记录 `knowledge_error`，避免「静默没有知识」被误认为「知识库是空的」。
"""

from __future__ import annotations

import time

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType
from ..rag import KnowledgeStore, KnowledgeStoreError
from ..tools import SqlToolError
from .base import AgentContext, BaseAgent, _Usage

_METRICS_SQL = (
    "SELECT metric_key, metric_name, definition, formula, owner_department "
    "FROM metric_definitions ORDER BY metric_key"
)


class RetrieverAgent(BaseAgent):
    role = "retriever"
    version = "0.2.0"

    async def _retrieve_knowledge(
        self, ctx: AgentContext, question: str
    ) -> tuple[list[dict], str | None]:
        """返回 (知识片段, 错误信息)。失败时返回空列表与错误原因，不抛异常。"""
        if not question.strip():
            return [], None
        try:
            chunks = KnowledgeStore(ctx.settings).search(question)
        except (KnowledgeStoreError, Exception) as exc:  # embedding 也可能失败
            return [], f"{type(exc).__name__}: {exc}"[:300]
        return [
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "score": round(c.score, 4),
                "vector_score": round(c.vector_score, 4),
                "keyword_score": round(c.keyword_score, 4),
            }
            for c in chunks
        ], None

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()  # 本节点不调用模型

        try:
            result = ctx.sandbox.execute(_METRICS_SQL)
            schema = ctx.sandbox.schema_as_prompt_text()
            # 数据实际覆盖的时间范围必须取出来给下游：否则取数环节会凭空假设年份，
            # 写出落在真实数据范围之外的日期条件，聚合结果全为 NULL。
            data_ranges = ctx.sandbox.data_ranges_as_prompt_text()
        except SqlToolError as exc:
            code, retryable = self.classify_error(exc)
            return self.failed(
                inv,
                code=code,
                message=str(exc)[:500],
                retryable=retryable,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        definitions = {
            row["metric_key"]: {
                "metric_name": row["metric_name"],
                "definition": row["definition"],
                "formula": row["formula"],
                "owner_department": row["owner_department"],
            }
            for row in result.rows
        }

        question = inv.input.get("question", "")
        knowledge, knowledge_error = await self._retrieve_knowledge(ctx, question)

        sources = [
            SourceRef(type=SourceType.sql, ref=f"sha256:{result.sql_hash}"),
            SourceRef(type=SourceType.document, ref="metric_definitions（指标口径库）"),
        ]
        sources.extend(
            SourceRef(type=SourceType.document, ref=f"知识库:{c['doc_id']}#{c['chunk_index']}")
            for c in knowledge
        )

        outputs: dict = {
            "metric_definitions": definitions,
            "schema": schema,
            "data_ranges": data_ranges,
            "metric_count": len(definitions),
            "knowledge": knowledge,
            "knowledge_count": len(knowledge),
        }
        if knowledge_error:
            # 显式记录：降级不等于没发生
            outputs["knowledge_error"] = knowledge_error

        return self.ok(
            inv,
            outputs=outputs,
            sources=sources,
            usage=usage,
            duration_ms=int((time.perf_counter() - started) * 1000),
            confidence=1.0 if definitions else 0.0,
            blackboard_writes=["retriever_1.metric_definitions"],
        )
