"""知识检索 Agent（Retriever）。

阶段一 W2 的职责：**取指标口径与数据字典**，为后续分析统一口径。
实现方式是直接查 `metric_definitions` 表（SQL 检索），不引入向量检索——
按《实施路线图》，语义检索与长期记忆（pgvector + Reranker）在 W3/W6 落地。

这里刻意不做一次额外的模型调用：口径数据是结构化的，直查比让模型转述更准、
更省 token，也避免模型在「口径」这种不允许失真的信息上引入幻觉。
"""

from __future__ import annotations

import time

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType
from ..tools import SqlToolError
from .base import AgentContext, BaseAgent, _Usage

_METRICS_SQL = (
    "SELECT metric_key, metric_name, definition, formula, owner_department "
    "FROM metric_definitions ORDER BY metric_key"
)


class RetrieverAgent(BaseAgent):
    role = "retriever"
    version = "0.1.0"

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()  # 本节点不调用模型

        try:
            result = ctx.sandbox.execute(_METRICS_SQL)
            schema = ctx.sandbox.schema_as_prompt_text()
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

        return self.ok(
            inv,
            outputs={
                "metric_definitions": definitions,
                "schema": schema,
                "metric_count": len(definitions),
            },
            sources=[
                SourceRef(
                    type=SourceType.sql,
                    ref=f"sha256:{result.sql_hash}",
                ),
                SourceRef(
                    type=SourceType.document,
                    ref="metric_definitions（指标口径库）",
                ),
            ],
            usage=usage,
            duration_ms=int((time.perf_counter() - started) * 1000),
            confidence=1.0 if definitions else 0.0,
            blackboard_writes=["retriever_1.metric_definitions"],
        )
