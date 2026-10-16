"""数据分析 Agent（Analyst）。

流程是三阶段，核心目标是「**把模型从编数字的路径上隔离出去**」：
1. **取数**：模型只把问题翻译成 SQL，不下结论；
2. **归因**：只把**沙箱真实执行的结果**喂回模型，且要求每个数字都来自结果。

关键设计（每一处都由评测基线暴露的问题驱动）：

- **规划阶段的验收标准会传给取数阶段**。首轮评测发现：SQL 只查了 9 月、没查 8 月
  做对比，导致归因阶段（被限制只能用结果里的真实数据）只能诚实拒绝。根因是
  「写 SQL 的一步不知道下结论需要什么」。
- **验收标准不传给归因阶段**。归因阶段若知道期望答案，会有迎合压力而编造数字；
  它的唯一输入是真实结果行。
- **数据真实时间范围会传给取数阶段**。评测中模型凭空假设年份，写出
  `stat_date='2025-08-01'` 这类范围外条件，聚合结果全为 NULL。
- **有界修补覆盖两类失败**：(a) SQL 被数据库拒绝（如 GROUP BY 与窗口函数冲突）；
  (b) 归因判定数据不足。两类都会把原因回灌、重新生成 SQL，最多重试到
  `MAX_SQL_ATTEMPTS` 次。真正的动态重规划属于 W7，这里只做有界修补。
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType
from ..gateway import ROUTE_ANALYST_CONCLUDE, ROUTE_ANALYST_SQL
from ..tools import SqlExecutionError, SqlResult
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
   存在其它年份或月份——写成范围外的日期会让聚合结果全为 NULL。
7. **SQL 语法必须能被 PostgreSQL 16 直接执行**：使用 GROUP BY 时，
   非聚合列必须出现在 GROUP BY 中；窗口函数不要与 GROUP BY 混用在非分组列上。
8. 下方可能给出「业务背景资料」。它可以帮助你判断**哪些查询才有意义**
   （例如资料说明数据是月度快照，就不要写日粒度的查询）。
9. 输出单个 JSON 对象，不要 markdown 围栏：
   {"sql": "<SQL 原文>", "rationale": "<为什么这样查，一句话>"}"""

_SQL_SYSTEM = f"""你是数据分析师的取数环节。你的唯一任务是：把业务问题翻译成一条**只读** SQL 查询。

{_SQL_RULES}"""

_SQL_REPAIR_SYSTEM = f"""你是数据分析师的取数环节。你上一轮生成的 SQL 没有通过（被数据库拒绝，或结果不足以支撑结论），
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
6. 下方可能给出「业务背景资料」。它**只能用于解释原因**（例如说明某段时间退货率
   上升是口径调整而非质量恶化），**严禁把其中的数字当作查询结果引用**——
   那是编造数据。引用背景资料时请注明出处文档标题。
7. 输出单个 JSON 对象，不要 markdown 围栏：
   {"summary": "...", "key_findings": ["...", "..."], "insufficient_data": false, "confidence": 0.8}"""


class _SqlPlan(BaseModel):
    sql: str = Field(min_length=1)
    rationale: str = ""


class _Conclusion(BaseModel):
    summary: str = Field(min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    insufficient_data: bool = False
    confidence: float = Field(default=0.5, ge=0, le=1)


def _knowledge_text(knowledge: list[dict], *, limit: int = 4, per_chunk: int = 400) -> str:
    """把检索到的知识片段压成提示词片段。截断是为控制输入长度与信噪比。"""
    if not knowledge:
        return "（无相关背景资料）"
    lines = []
    for item in knowledge[:limit]:
        content = str(item.get("content", "")).replace("\n", " ")[:per_chunk]
        lines.append(f"《{item.get('title', '?')}》片段{item.get('chunk_index', 0)}：{content}")
    return "\n".join(lines)


class AnalystAgent(BaseAgent):
    role = "analyst"
    version = "0.3.0"

    # 初次 + 至多一次修补。
    # 该上限由 20 条评测基线定：两类失败（SQL 被拒绝、归因判定数据不足）
    # 都在**下一次尝试**即被修复；设到 3 次时唯一效果是造出最差离群点
    # （biggest_contributor 166.6 秒 / 40,600 token），成功率并未提高。
    # 放宽重试属于 W7 的异常处理范畴，且需配合任务级 Token 熔断。
    MAX_SQL_ATTEMPTS = 2

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
            ctx,
            system=system,
            user=user,
            model_cls=_SqlPlan,
            usage=usage,
            route=ROUTE_ANALYST_SQL,
        )

    async def _conclude(
        self,
        ctx: AgentContext,
        *,
        usage: _Usage,
        question: str,
        sql: str,
        rows_text: str,
        row_count: int,
        knowledge_text: str,
    ) -> _Conclusion:
        return await self.call_json(
            ctx,
            system=_CONCLUDE_SYSTEM,
            user=(
                f"业务问题：{question}\n\n"
                f"实际执行的 SQL：\n{sql}\n\n"
                f"查询结果（共 {row_count} 行）：\n{rows_text}\n\n"
                f"业务背景资料（仅可用于解释原因，其中的数字不得当作查询结果）：\n"
                f"{knowledge_text}"
            ),
            model_cls=_Conclusion,
            usage=usage,
            route=ROUTE_ANALYST_CONCLUDE,
        )

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()

        question = inv.input.get("question", "")
        schema = inv.input.get("schema", "")
        data_ranges = inv.input.get("data_ranges", "")
        definitions = inv.input.get("metric_definitions", {})
        knowledge = inv.input.get("knowledge", []) or []
        now = inv.input.get("now", "")
        acceptance: list[str] = inv.input.get("acceptance", []) or []

        defs_text = "\n".join(
            f"- {key}（{d.get('metric_name', '')}）：{d.get('definition', '')} "
            f"公式：{d.get('formula', '')}"
            for key, d in definitions.items()
        ) or "（无口径数据）"
        acceptance_text = "\n".join(f"- {a}" for a in acceptance)
        knowledge_text = _knowledge_text(knowledge)

        base_user = (
            f"当前时间：{now}\n\n"
            f"可用表结构：\n{schema}\n\n"
            f"**数据实际覆盖范围（日期条件必须落在此范围内）**：\n"
            f"{data_ranges or '（未取到，请先用可用范围内的日期）'}\n\n"
            f"指标口径：\n{defs_text}\n\n"
            f"业务背景资料（可帮助判断哪些查询才有意义）：\n{knowledge_text}\n\n"
            f"业务问题：{question}"
        )

        best: tuple[_SqlPlan, SqlResult, _Conclusion] | None = None
        feedback: str | None = None
        attempts = 0
        problems: list[str] = []

        try:
            while attempts < self.MAX_SQL_ATTEMPTS:
                attempts += 1
                try:
                    plan = await self._generate_sql(
                        ctx,
                        usage=usage,
                        base_user=base_user,
                        acceptance_text=acceptance_text,
                        repair_feedback=feedback,
                    )
                    result = ctx.sandbox.execute(plan.sql)
                except SqlExecutionError as exc:
                    problems.append(f"第{attempts}次 SQL 被拒绝: {str(exc)[:200]}")
                    if best is not None:
                        break  # 已有可用结果，不再为报错消耗额度
                    feedback = (
                        f"上一轮 SQL 被数据库拒绝，报错如下：\n{str(exc)[:400]}\n"
                        "请修正。注意：使用 GROUP BY 时非聚合列必须出现在 GROUP BY 中。"
                    )
                    continue

                conclusion = await self._conclude(
                    ctx,
                    usage=usage,
                    question=question,
                    sql=result.sql,
                    rows_text=result.as_prompt_text(),
                    row_count=result.row_count,
                    knowledge_text=knowledge_text,
                )
                if best is None or not conclusion.insufficient_data:
                    best = (plan, result, conclusion)
                if not conclusion.insufficient_data:
                    break

                problems.append(f"第{attempts}次数据不足: {conclusion.summary[:150]}")
                feedback = (
                    f"{conclusion.summary}\n上一轮 SQL：\n{plan.sql}\n"
                    f"上一轮结果（{result.row_count} 行）：\n{result.as_prompt_text(20)}"
                )

            if best is None:
                raise SqlExecutionError("；".join(problems) or "所有 SQL 尝试均未成功")
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

        plan, result, conclusion = best
        return self.ok(
            inv,
            outputs={
                "sql": result.sql,
                "sql_hash": result.sql_hash,
                "sql_rationale": plan.rationale,
                "sql_attempts": attempts,
                "repair_problems": problems,
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
            # 黑板 key 由实际 node_id 派生：动态 DAG 下节点名是运行时可变的，
            # 写死 "analyst_1.report" 会让多个 analyst 节点互相覆盖。
            blackboard_writes=[f"{inv.node_id}.report"],
        )
