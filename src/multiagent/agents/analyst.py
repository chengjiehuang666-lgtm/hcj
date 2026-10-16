"""数据分析 Agent（Analyst）。

两阶段执行，这是「结论可回溯」的关键设计：
1. **取数**：模型只负责把问题翻译成 SQL，不负责下结论；
2. **归因**：把**真实执行结果**喂回模型，要求结论必须引用其中的数字。

这样把模型从「凭记忆编数字」的路径上隔离出去——它只能基于沙箱跑出来的
真实行做推理。SQL 原文与哈希随结果一并返回，用于溯源。
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType
from .base import AgentContext, BaseAgent, _Usage

_SQL_SYSTEM = """你是数据分析师的取数环节。你的唯一任务是：把业务问题翻译成一条**只读** SQL 查询。

硬性要求：
1. 只允许 SELECT 或 WITH 开头的单条语句；禁止任何写操作、DDL、多语句。
2. 只能使用下面列出的表和列，不要臆造表名或列名。
3. 优先用聚合（SUM/COUNT/GROUP BY）产出**小而清晰**的结果集，行数控制在 50 行以内。
4. 需要环比/同比时，用条件聚合或按日期分组，不要写复杂嵌套。
5. 输出单个 JSON 对象，不要 markdown 围栏：
   {"sql": "<SQL 原文>", "rationale": "<为什么这样查，一句话>"}"""

_CONCLUDE_SYSTEM = """你是资深数据分析师，负责基于**给定查询结果**给出结论。

硬性要求：
1. 结论中的每一个数字都必须来自下面给出的查询结果，**严禁自行编造或估算数字**。
2. 必须给出归因：指出下滑/变化主要来自哪个维度（区域/渠道/品类），并给出具体幅度。
3. key_findings 每条都要带具体数字，例如「华东区线上数码销售额环比下降 37.6%」。
4. 若查询结果不足以支撑结论，把 insufficient_data 设为 true，并在 summary 说明还缺什么数据。
5. confidence 是 0-1 的数字，表示你对结论的把握程度。
6. 输出单个 JSON 对象，不要 markdown 围栏：
   {"summary": "...", "key_findings": ["...", "..."], "insufficient_data": false, "confidence": 0.8}"""


class _SqlPlan(BaseModel):
    sql: str = Field(min_length=1)
    rationale: str = ""


class _Conclusion(BaseModel):
    summary: str = Field(min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    insufficient_data: bool = False
    confidence: float = Field(default=0.5, ge=0, le=1)


class AnalystAgent(BaseAgent):
    role = "analyst"
    version = "0.1.0"

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()

        question = inv.input.get("question", "")
        schema = inv.input.get("schema", "")
        definitions = inv.input.get("metric_definitions", {})
        now = inv.input.get("now", "")

        defs_text = "\n".join(
            f"- {key}（{d.get('metric_name','')}）：{d.get('definition','')} 公式：{d.get('formula','')}"
            for key, d in definitions.items()
        ) or "（无口径数据）"

        try:
            # ---- 阶段一：生成 SQL ----
            plan = await self.call_json(
                ctx,
                system=_SQL_SYSTEM,
                user=(
                    f"当前时间：{now}\n\n可用表结构：\n{schema}\n\n"
                    f"指标口径：\n{defs_text}\n\n业务问题：{question}"
                ),
                model_cls=_SqlPlan,
                usage=usage,
            )

            # ---- 阶段二：在沙箱里真实执行 ----
            result = ctx.sandbox.execute(plan.sql)

            # ---- 阶段三：基于真实结果归因 ----
            conclusion = await self.call_json(
                ctx,
                system=_CONCLUDE_SYSTEM,
                user=(
                    f"业务问题：{question}\n\n"
                    f"实际执行的 SQL：\n{result.sql}\n\n"
                    f"查询结果（共 {result.row_count} 行）：\n{result.as_prompt_text()}"
                ),
                model_cls=_Conclusion,
                usage=usage,
            )
        except Exception as exc:
            code, retryable = self.classify_error(exc)
            return self.failed(
                inv,
                code=code,
                message=str(exc)[:500],
                retryable=retryable,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        return self.ok(
            inv,
            outputs={
                "sql": result.sql,
                "sql_hash": result.sql_hash,
                "sql_rationale": plan.rationale,
                "row_count": result.row_count,
                "rows_preview": result.rows[:50],
                "result_limited": result.limited,
                "summary": conclusion.summary,
                "key_findings": conclusion.key_findings,
                "insufficient_data": conclusion.insufficient_data,
            },
            sources=[
                SourceRef(type=SourceType.sql, ref=f"sha256:{result.sql_hash}"),
                SourceRef(type=SourceType.model_inference, ref=self.labeled_version),
            ],
            usage=usage,
            duration_ms=int((time.perf_counter() - started) * 1000),
            confidence=conclusion.confidence,
            blackboard_writes=["analyst_1.report"],
        )
