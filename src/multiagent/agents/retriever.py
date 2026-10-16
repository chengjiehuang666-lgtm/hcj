"""知识检索 Agent（Retriever）。

职责：为后续分析准备上下文。支持四种模式，由**节点自带的 `inputs.mode`** 指定——
编排层不再为每个节点写死行为，节点读自己的声明来决定做什么（这正是动态 DAG 的意义）：

| mode | 内容 | 用途 |
| :--- | :--- | :--- |
| `metrics` | 指标口径 + 表结构 + 数据时间范围（走 SQL） | 与 `knowledge` 并行的一路 |
| `knowledge` | 知识库混合检索：政策、历史复盘、数据口径注意事项 | 与 `metrics` 并行的一路 |
| `memory` | 长期记忆混合检索：历次任务的结论 | 问题涉及「上次/回顾/承接」时的一路 |
| `all` | metrics + knowledge（默认，向后兼容单节点形态；**不含 memory**） | 旧链路 / 简单任务 |

设计取舍：
- 口径数据刻意**不让模型转述**——结构化数据直查更准、更省 token，
  也避免模型在「口径」这种不允许失真的信息上引入幻觉；
- 知识库检索同理不经过模型：embedding + SQL 加权排序即可，交给模型转述只会失真；
- **知识/记忆检索失败不让节点失败**：它们是增强而非必需，检索不到就降级，
  但在结果里显式记录 `knowledge_error` / `memory_error`，避免「静默没有」被误认为
  「库是空的」。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType
from ..memory import MemoryStoreError, MemoryStore
from ..rag import KnowledgeStore, KnowledgeStoreError
from ..tools import SqlToolError
from .base import AgentContext, BaseAgent, _Usage

_METRICS_SQL = (
    "SELECT metric_key, metric_name, definition, formula, owner_department "
    "FROM metric_definitions ORDER BY metric_key"
)

MODE_METRICS = "metrics"
MODE_KNOWLEDGE = "knowledge"
MODE_MEMORY = "memory"
MODE_ALL = "all"
VALID_MODES = (MODE_METRICS, MODE_KNOWLEDGE, MODE_MEMORY, MODE_ALL)


class RetrieverAgent(BaseAgent):
    role = "retriever"
    version = "0.3.0"

    async def _retrieve_knowledge(
        self, ctx: AgentContext, question: str
    ) -> tuple[list[dict], str | None]:
        """返回 (知识片段, 错误信息)。失败时返回空列表与错误原因，不抛异常。"""
        if not question.strip():
            return [], None
        try:
            # 走线程：embedding（CPU 推理）与向量检索（同步 psycopg）都是阻塞调用，
            # 直接在事件循环里跑会占住循环，让本该并行的另一路检索排队等待。
            chunks = await asyncio.to_thread(
                lambda: KnowledgeStore(ctx.settings).search(question)
            )
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

    async def _retrieve_memory(
        self, ctx: AgentContext, question: str
    ) -> tuple[list[dict], str | None]:
        """长期记忆检索：返回 (命中条目, 错误信息)。失败降级，不抛异常。"""
        if not question.strip():
            return [], None
        try:
            hits = await asyncio.to_thread(
                lambda: MemoryStore(ctx.settings).search(question)
            )
        except (MemoryStoreError, Exception) as exc:  # embedding 也可能失败
            return [], f"{type(exc).__name__}: {exc}"[:300]
        return [
            {
                "memory_id": h.memory_id,
                "kind": h.kind,
                "content": h.content,
                "source_ref": h.source_ref,
                "confidence": h.confidence,
                "created_at": h.created_at,
                "score": round(h.score, 4),
            }
            for h in hits
        ], None

    def _metrics_part(self, ctx: AgentContext) -> tuple[dict[str, Any], list[SourceRef]]:
        """走 SQL 的那一路。失败抛 SqlToolError，由调用方转成 AgentResult(failed)。"""
        result = ctx.sandbox.execute(_METRICS_SQL)
        definitions = {
            row["metric_key"]: {
                "metric_name": row["metric_name"],
                "definition": row["definition"],
                "formula": row["formula"],
                "owner_department": row["owner_department"],
            }
            for row in result.rows
        }
        outputs: dict[str, Any] = {
            "metric_definitions": definitions,
            "metric_count": len(definitions),
            "schema": ctx.sandbox.schema_as_prompt_text(),
            # 数据实际覆盖的时间范围必须给下游：否则取数环节会凭空假设年份，
            # 写出落在真实数据范围之外的日期条件，聚合结果全为 NULL。
            "data_ranges": ctx.sandbox.data_ranges_as_prompt_text(),
        }
        sources = [
            SourceRef(type=SourceType.sql, ref=f"sha256:{result.sql_hash}"),
            SourceRef(type=SourceType.document, ref="metric_definitions（指标口径库）"),
        ]
        return outputs, sources

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()  # 本节点不调用模型

        # 读节点自带的 inputs.mode —— 键名必须与 Planner 提示词里声明的一致。
        # （早先这里写成 retrieval_mode、而 Planner 写 mode，导致两个检索节点
        #   都退化成 all 模式重复检索；单测没抓到是因为测试与实现用了同一个错键名。）
        mode = str(inv.input.get("mode") or MODE_ALL).lower()
        if mode not in VALID_MODES:
            return self.failed(
                inv,
                code="BAD_INPUT",
                message=f"未知的检索模式 {mode!r}；可用: {list(VALID_MODES)}",
                retryable=False,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        outputs: dict[str, Any] = {}
        sources: list[SourceRef] = []
        blackboard_writes: list[str] = []
        # 注意：不要把 mode 写进 outputs。它是**输入声明**而非产出，
        # 写进去会在下游合并上游产物时产生同名键冲突（无害但脏）。

        if mode in (MODE_METRICS, MODE_ALL):
            try:
                # 同样走线程：schema 自省、数据范围查询都是同步阻塞的 psycopg 调用。
                # 不这么做的话，「两路检索并行」在实测里会退化成串行——
                # 事件时间线显示 retriever_2 是等 retriever_1 结束才启动的。
                metrics_outputs, metrics_sources = await asyncio.to_thread(
                    self._metrics_part, ctx
                )
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
            outputs.update(metrics_outputs)
            sources.extend(metrics_sources)
            blackboard_writes.append(f"{inv.node_id}.metrics")

        if mode in (MODE_KNOWLEDGE, MODE_ALL):
            question = inv.input.get("question", "")
            knowledge, knowledge_error = await self._retrieve_knowledge(ctx, question)
            outputs["knowledge"] = knowledge
            outputs["knowledge_count"] = len(knowledge)
            if knowledge_error:
                # 显式记录：降级不等于没发生
                outputs["knowledge_error"] = knowledge_error
            sources.extend(
                SourceRef(
                    type=SourceType.document,
                    ref=f"知识库:{c['doc_id']}#{c['chunk_index']}",
                )
                for c in knowledge
            )
            if not knowledge:
                # 检索无命中**也是结果**，必须留下来源记录。
                # 契约要求 status=ok 的结果必须带 sources，而「没命中」时上面那段
                # 一条都加不上 → 会在 ok() 处触发契约校验失败，让「知识库无相关内容」
                # 这种完全正常的情况变成节点崩溃（曾经就是这样）。
                sources.append(
                    SourceRef(
                        type=SourceType.tool,
                        ref=f"知识库检索无命中（查询：{question[:40]}）",
                    )
                )
            blackboard_writes.append(f"{inv.node_id}.knowledge")

        if mode == MODE_MEMORY:
            question = inv.input.get("question", "")
            memories, memory_error = await self._retrieve_memory(ctx, question)
            outputs["memories"] = memories
            outputs["memory_count"] = len(memories)
            if memory_error:
                outputs["memory_error"] = memory_error
            sources.extend(
                SourceRef(type=SourceType.document, ref=f"长期记忆:{m['memory_id']}")
                for m in memories
            )
            if not memories:
                # 与知识库同口径：「没命中」也是结果，必须留下来源记录
                sources.append(
                    SourceRef(
                        type=SourceType.tool,
                        ref=f"长期记忆检索无命中（查询：{question[:40]}）",
                    )
                )
            blackboard_writes.append(f"{inv.node_id}.memory")

        # 置信度按「本模式期望的内容是否取到」判定
        got_metrics = bool(outputs.get("metric_definitions"))
        got_knowledge = bool(outputs.get("knowledge"))
        got_memories = bool(outputs.get("memories"))
        if mode == MODE_METRICS:
            confidence = 1.0 if got_metrics else 0.0
        elif mode == MODE_KNOWLEDGE:
            confidence = 1.0 if got_knowledge else 0.0
        elif mode == MODE_MEMORY:
            confidence = 1.0 if got_memories else 0.0
        else:
            confidence = 1.0 if (got_metrics or got_knowledge) else 0.0

        return self.ok(
            inv,
            outputs=outputs,
            sources=sources,
            usage=usage,
            duration_ms=int((time.perf_counter() - started) * 1000),
            confidence=confidence,
            blackboard_writes=blackboard_writes,
        )
