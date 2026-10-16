"""数据分析 Agent（Analyst）。

三阶段执行，核心是「**把模型从编数字的路径上隔离出去**」：
1. **取数**：模型只把问题翻译成 SQL，不下结论；
2. **归因**：只把**沙箱真实执行的结果**喂回模型，且要求每个数字都来自结果。

关键设计（由评测基线暴露的问题驱动）：
- 规划阶段的**验收标准会传给取数阶段**。首轮评测发现：SQL 只查了 9 月、
  没查 8 月做对比，导致归因阶段（被限制只能用结果里的真实数据）只能诚实拒绝。
  根因是「写 SQL 的一步不知道下结论需要什么」，故把验收标准透传下来。
- **验收标准不传给归因阶段**。归因阶段若知道期望答案，会有迎合压力而编造数字；
  它的唯一输入是真实结果行。
- **不足即修补**：若归因判定数据不足，允许**重写一次 SQL** 再试（最多一次）。
  真正的动态重规划属于 W7，这里只做一次有界修补。
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType
from .base import AgentContext, BaseAgent, _Usage

_SQL_RULES = """硬性要求：
1. 只允许 SELECT 或 WITH 开头的单条语句；禁止任何写操作、DDL、多语句。
2. 只能使用下面列出的表和列，不要臆造表名或列名。
3. 用聚合（SUM/COUNT/GROUP BY）产出**小而清晰**的结果集，行数控制在 50 行以内。
4. **若问题涉及变化、下滑、对比、环比、趋势，查询结果必须同时包含对比期与基期的数据。**
   宁可按维度或月份分组返回多行，也不要只返回一个孤立的汇总值——
   下游只会基于你返回的结果做判断，缺少基期数据将导致无法给出结论。
5. 查询结果要足以支撑下面列出的「验收标准」，每条都要有对应数据可用。
6. **日期条件必须落在「数据实际覆盖范围」之内**（见下方说明）。不得假设数据里
   存在其它年份或月份——写成范围外的日期会让聚合结果全为 NULL，导致无法给出结论。
7. 输出单个 JSON 对象，不要 markdown 围栏：
   {"sql": "<SQL 原文>", "rationale": "<为什么这样查，一句话>"}"""

_SQL_SYSTEM = f"""你是数据分析师的取数环节。你的唯一任务是：把业务问题翻译成一条**只读** SQL 查询。

{_SQL_RULES}"""

_SQL_REPAIR_SYSTEM = f"""你是数据分析师的取数环节。你上一轮生成的 SQL 被判定为**结果不足以支撑结论**，
请根据反馈重写。

{_SQL_RULES}"""

_CONCLUDE_SYSTEM = """你是资深数据分析师，负责基于**给定查询结果**给出结论。

硬性要求：
1. 结论中的每一个数字都必须来自下面给出的查询结果，**严禁自行编造或估算数字**。
2. 必须给出归因：指出下滑/变化主要来自哪个维度（区域/渠道/品类），并给出具体幅度。
3. key_findings 每条都要带具体数字，例如「华东区线上数码销售额环比下降 37.6%」。
4. 若查询结果确实不足以支撑结论，把 insufficient_data 设为 true，
   并在 summary 中**明确说明还缺什么数据**（例如「缺少 8 月基期数据」），不要硬凑结论。
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
    version = "0.2.0"

    MAX_SQL_ATTEMPTS = 2  # 初次 + 至多一次修补

    async def _generate_sql(
        self,
        ctx: AgentContext,
        *,
        usage: _Usage,
        base_user: str,
        acceptance_text: str,
        repair_feedback: str | None = None,
    ) -> _SqlPlan:
        system = _SQL_REPAIR_SYSTEM if repair_feedback else _SQL_SYSTEM
        user = base_user
        if acceptance_text:
            user += f"\n\n本次需要支撑的验收标准：\n{acceptance_text}"
        if repair_feedback:
            user += f"\n\n上一轮的问题反馈：\n{repair_feedback}"
        return await self.call_json(
            ctx, system=system, user=user, model_cls=_SqlPlan, usage=usage
        )

    async def _conclude(
        self, ctx: AgentContext, *, usage: _Usage, question: str, sql: str, rows_text: str, row_count: int
    ) -> _Conclusion:
        return await self.call_json(
            ctx,
            system=_CONCLUDE_SYSTEM,
            user=(
                f"业务问题：{question}\n\n"
                f"实际执行的 SQL：\n{sql}\n\n"
                f"查询结果（共 {row_count} 行）：\n{rows_text}"
            ),
            model_cls=_Conclusion,
            usage=usage,
        )

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()

        question = inv.input.get("question", "")
        schema = inv.input.get("schema", "")
        data_ranges = inv.input.get("data_ranges", "")
        definitions = inv.input.get("metric_definitions", {})
        now = inv.input.get("now", "")
        acceptance: list[str] = inv.input.get("acceptance", []) or []

        defs_text = "\n".join(
            f"- {key}（{d.get('metric_name', '')}）：{d.get('definition', '')} "
            f"公式：{d.get('formula', '')}"
            for key, d in definitions.items()
        ) or "（无口径数据）"
        acceptance_text = "\n".join(f"- {a}" for a in acceptance)

        base_user = (
            f"当前时间：{now}\n\n"
            f"可用表结构：\n{schema}\n\n"
            f"**数据实际覆盖范围（日期条件必须落在此范围内）**：\n"
            f"{data_ranges or '（未取到，请先用可用范围内的日期）'}\n\n"
            f"指标口径：\n{defs_text}\n\n业务问题：{question}"
        )

        try:
            repair_feedback: str | None = None
            plan = await self._generate_sql(
                ctx,
                usage=usage,
                base_user=base_user,
                acceptance_text=acceptance_text,
            )
            result = ctx.sandbox.execute(plan.sql)
            conclusion = await self._conclude(
                ctx,
                usage=usage,
                question=question,
                sql=result.sql,
                rows_text=result.as_prompt_text(),
                row_count=result.row_count,
            )

            # 数据不足时做一次有界修补：把「缺什么」回灌给取数环节重写 SQL
            sql_attempts = 1
            if conclusion.insufficient_data and self.MAX_SQL_ATTEMPTS > 1:
                repair_feedback = (
                    f"{conclusion.summary}\n"
                    f"上一轮 SQL：\n{plan.sql}\n"
                    f"上一轮结果（{result.row_count} 行）：\n{result.as_prompt_text(20)}"
                )
                plan2 = await self._generate_sql(
                    ctx,
                    usage=usage,
                    base_user=base_user,
                    acceptance_text=acceptance_text,
                    repair_feedback=repair_feedback,
                )
                result2 = ctx.sandbox.execute(plan2.sql)
                conclusion2 = await self._conclude(
                    ctx,
                    usage=usage,
                    question=question,
                    sql=result2.sql,
                    rows_text=result2.as_prompt_text(),
                    row_count=result2.row_count,
                )
                sql_attempts = 2
                # 修补后仍不足则保留修补结果；若补上了则采用新结论
                if not conclusion2.insufficient_data or result2.row_count > result.row_count:
                    plan, result, conclusion = plan2, result2, conclusion2
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
                "sql_attempts": sql_attempts,
                "row_count": result.row_count,
                "rows_preview": result.rows[:50],
                "result_limited": result.limited,
                "summary": conclusion.summary,
                "key_findings": conclusion.key_findings,
                "insufficient_data": conclusion.insufficient_data,
                "acceptance": acceptance,
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
