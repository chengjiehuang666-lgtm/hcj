"""评测执行器：跑评测集、打分、产出 D2 与 M1 的基线数据。

设计要点：
- **串行执行**（并发=1）。时延基线必须是「单任务真实时延」，并发跑出来的延迟
  没有参考意义。
- **打分不靠模型自评**，而是拿数据库实测算出的已知事实去核对结论文本。
- 数字匹配做了多种写法归一：`2,335,000`、`2335000`、`233.5万` 视为同一事实。

用法：
    python -m multiagent.eval.runner --limit 3          # 冒烟
    python -m multiagent.eval.runner                    # 全量 20 条
    python -m multiagent.eval.runner --ids core_decline_cause,east_refund_rate
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from ..asyncutil import run_async
from ..config import Settings
from ..graph import AnalysisPipeline
from ..graph.pipeline import TZ_CN, new_task_event
from .cases import CASES, EvalCase, get_cases


# --------------------------------------------------------------------- 打分


def normalize(text: str) -> str:
    """去掉逗号与空白，使 2,335,000 / 2335000 等价。"""
    for ch in (",", "，", " ", "\u00a0", "\t", "\n"):
        text = text.replace(ch, "")
    return text


def _alternatives(needle: str) -> list[str]:
    """为纯数字期望值补充「万」写法，避免模型用 295万 而判失败。"""
    if not needle.isdigit():
        return [needle]
    value = int(needle)
    alts = {needle}
    if value >= 10_000:
        for digits in (0, 1, 2):
            alts.add(f"{value / 10_000:.{digits}f}万")
    return sorted(alts)


def _matches(haystack: str, needle: str) -> bool:
    return any(alt in haystack for alt in _alternatives(needle))


@dataclass
class CaseResult:
    case_id: str
    question: str
    passed: bool
    status: str
    latency_ms: int
    tokens: int
    input_tokens: int
    output_tokens: int
    failed_checks: list[str] = field(default_factory=list)
    hit_forbidden: list[str] = field(default_factory=list)
    insufficient_data: bool = False
    sql_rows: int | None = None
    sql_hash: str | None = None
    summary: str = ""
    error: str | None = None
    #: 按调用点（planner / analyst_sql / analyst_conclude）的成本与时延归因。
    #: 分级路由靠它决定哪些调用点值得切到 fast 档。
    route_tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    #: 检索阶段实际取到的知识文档与片段数——RAG 回归的结构性证据
    retrieved_docs: list[str] = field(default_factory=list)
    knowledge_count: int = 0
    #: 评审结论（verdict 与问题数）——reviewer 的可观测出口
    review_verdict: str | None = None
    review_issues: int | None = None
    #: 检索阶段实际取到的长期记忆条数——记忆回归的结构性证据
    memory_count: int = 0
    #: 本次任务是否真的沉淀进长期记忆（None=未沉淀/不适用）；失败原因
    memory_sunk: int | None = None
    memory_error: str | None = None


def _nodes_by_role(outcome, role: str) -> list[dict]:
    """按 agent_role 取节点结果。

    动态 DAG 下节点 ID 是运行时可变的，**按 role 取值才稳定**——
    原先写死的 `node_id == "analyst_1"` 在 W5 之后会全线失效。
    这里直接遍历 node_results（而非调用 PipelineOutcome 的方法），
    是为了让测试用的假 outcome 也能工作。
    """
    return [r for r in outcome.node_results if r.get("agent_role") == role]


def _analyst_outputs(outcome) -> dict:
    """分析节点的产物（多个 analyst 节点时按顺序合并，后者覆盖同名键）。"""
    merged: dict = {}
    for res in _nodes_by_role(outcome, "analyst"):
        merged.update(res.get("outputs") or {})
    return merged


def _retrieved_docs(outcome) -> list[str]:
    """汇总**所有** retriever 节点取到的知识文档。

    检索拆成两路并行后，knowledge 只可能出现在其中一路（mode=knowledge），
    因此必须遍历全部 retriever 节点，不能只看第一个。
    """
    docs: list[str] = []
    for res in _nodes_by_role(outcome, "retriever"):
        for item in (res.get("outputs") or {}).get("knowledge", []) or []:
            docs.append(str(item.get("doc_id", "")))
    return docs


def _retrieved_memories(outcome) -> list[str]:
    """汇总所有 retriever 节点取到的长期记忆内容（结构性命中证据）。"""
    texts: list[str] = []
    for res in _nodes_by_role(outcome, "retriever"):
        for item in (res.get("outputs") or {}).get("memories", []) or []:
            texts.append(str(item.get("content", "")))
    return texts


def _writer_outputs(outcome) -> dict:
    """写作节点的产物（多节点时按顺序合并）。"""
    merged: dict = {}
    for res in _nodes_by_role(outcome, "writer"):
        merged.update(res.get("outputs") or {})
    return merged


def _memory_source_problem(case: EvalCase, results: list[CaseResult]) -> str | None:
    """记忆用例的前置条件：它依赖的前置用例必须本次**真的沉淀**。

    这条断言必须查前置用例的 `memory_sunk`，而不是当前用例召回内容的来源——
    召回是否包含某条特定记忆取决于**检索排名**（相似历史记忆一多就不保证进
    top-k），那是产品没承诺的性质；而「沉淀链路是否工作」是本次交付必须守住的。
    """
    if not case.memory_source_case:
        return None
    producer = next(
        (r for r in results if r.case_id == case.memory_source_case), None
    )
    if producer is None:
        return f"前置用例未运行:{case.memory_source_case}"
    if producer.memory_sunk is None:
        return f"前置用例未沉淀:{case.memory_source_case}"
    return None


def _review_summary(outcome) -> tuple[str | None, int | None]:
    """取 reviewer 节点的 verdict 与问题数（无 reviewer 节点时返回 None）。"""
    reviews = _nodes_by_role(outcome, "reviewer")
    if not reviews:
        return None, None
    outputs: dict = {}
    for res in reviews:
        outputs.update(res.get("outputs") or {})
    return outputs.get("verdict"), outputs.get("issue_count")


def grade(case: EvalCase, outcome) -> tuple[bool, list[str], list[str], bool]:
    """返回 (是否通过, 未命中的检查项, 命中的禁用词, 是否判定数据不足)。

    注意：记忆用例「前置用例真的沉淀了」这一断言不在这里——沉淀结果挂在**前置
    用例**的 CaseResult 上，由 run_cases 校验（见 `_memory_source_problem`）。
    此前试过把断言绑成「召回里要有前置用例本次 task_id」：它依赖检索排名，库里
    相似历史记忆一多就不保证进 top-k，属于把产品没承诺的排序性质当门槛。
    """
    outputs = _analyst_outputs(outcome)
    insufficient = bool(outputs.get("insufficient_data"))

    text = normalize(
        " ".join(
            [
                outcome.summary or "",
                " ".join(outputs.get("key_findings", []) or []),
                # 文档场景：产物本体是文档正文，断言必须能落在它上面
                str(_writer_outputs(outcome).get("document_markdown") or "")[:3000],
            ]
        )
    )

    if outcome.status != "ok":
        return False, ["pipeline_failed"], [], insufficient

    failed: list[str] = []
    if case.expect_insufficient and insufficient:
        pass  # 模型已显式声明数据不足，视为满足该用例的核心要求
    else:
        for group in case.expect_any:
            if not any(_matches(text, alt) for alt in group):
                failed.append("|".join(group))

    # 角色证据：该能力真的参与了执行（如文档场景必须有 writer 且成功）
    ok_roles = {
        r.get("agent_role") for r in outcome.node_results if r.get("status") == "ok"
    }
    for role in case.expect_roles:
        if role not in ok_roles:
            failed.append(f"缺少角色:{role}")

    # RAG 回归：知识必须**真的被检索到**。
    # 只检查答案文本无法区分「模型本来就知道」与「查了知识库」，
    # 因此这里查的是检索结果这一结构性证据。
    docs = _retrieved_docs(outcome)
    if case.require_knowledge and not docs:
        failed.append("未检索到任何知识")
    if case.expect_docs and not any(exp in doc for exp in case.expect_docs for doc in docs):
        failed.append("知识库:" + "|".join(case.expect_docs))

    # 记忆回归：与 RAG 同一套方法——只查答案文本区分不出
    # 「真查了记忆」与「重新跑一遍 SQL 恰好得到相同数字」。
    memories = _retrieved_memories(outcome)
    if case.expect_memories and not any(
        exp in t for exp in case.expect_memories for t in memories
    ):
        failed.append("记忆:" + "|".join(case.expect_memories))

    hit_forbidden = [w for w in case.forbid if _matches(text, w)]
    return (not failed and not hit_forbidden), failed, hit_forbidden, insufficient


# --------------------------------------------------------------------- 执行


@dataclass
class EvalReport:
    started_at: str
    model: str
    provider: str
    total_cases: int
    passed: int
    results: list[CaseResult] = field(default_factory=list)
    wall_time_s: float = 0.0

    # ---- 汇总 ----
    @property
    def success_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases else 0.0

    def _latencies(self) -> list[float]:
        return [r.latency_ms / 1000 for r in self.results if r.status == "ok"]

    def _tokens(self) -> list[int]:
        return [r.tokens for r in self.results if r.status == "ok"]

    @property
    def p50_latency_s(self) -> float:
        data = self._latencies()
        return statistics.median(data) if data else 0.0

    @property
    def p95_latency_s(self) -> float:
        data = sorted(self._latencies())
        if not data:
            return 0.0
        index = min(len(data) - 1, int(round(0.95 * (len(data) - 1))))
        return data[index]

    @property
    def max_latency_s(self) -> float:
        data = self._latencies()
        return max(data) if data else 0.0

    @property
    def mean_tokens(self) -> float:
        data = self._tokens()
        return statistics.fmean(data) if data else 0.0

    @property
    def max_tokens(self) -> int:
        data = self._tokens()
        return max(data) if data else 0

    @property
    def p95_tokens(self) -> int:
        data = sorted(self._tokens())
        if not data:
            return 0
        return data[min(len(data) - 1, int(round(0.95 * (len(data) - 1))))]

    def route_summary(self) -> dict[str, dict[str, int]]:
        """跨全部用例，按调用点汇总次数/token/耗时。"""
        summary: dict[str, dict[str, int]] = {}
        for result in self.results:
            for route, stats in (result.route_tokens or {}).items():
                bucket = summary.setdefault(
                    route,
                    {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "duration_ms": 0,
                    },
                )
                for key in bucket:
                    bucket[key] += stats.get(key, 0)
        return summary


async def run_cases(settings: Settings, cases: list[EvalCase]) -> EvalReport:
    pipeline = AnalysisPipeline(settings)
    report = EvalReport(
        started_at=datetime.now(TZ_CN).isoformat(timespec="seconds"),
        model=settings.primary.model,
        provider=settings.primary.provider,
        total_cases=len(cases),
        passed=0,
    )

    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.id} … ", end="", flush=True)
        event = new_task_event(case.question)
        try:
            outcome = await pipeline.run(event)
        except Exception as exc:  # 单条用例失败不应中断整轮评测
            # 记录完整栈：只留一句异常消息会让间歇性崩溃无从排查
            import traceback

            report.results.append(
                CaseResult(
                    case_id=case.id,
                    question=case.question,
                    passed=False,
                    status="crashed",
                    latency_ms=0,
                    tokens=0,
                    input_tokens=0,
                    output_tokens=0,
                    failed_checks=["exception"],
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"[:3000],
                )
            )
            print("崩溃")
            continue

        passed, failed, forbidden, insufficient = grade(case, outcome)
        # 记忆用例：前置用例必须**真的沉淀**（沉淀结果在它的 CaseResult 上）
        if problem := _memory_source_problem(case, report.results):
            failed.append(problem)
            passed = False
        outputs = _analyst_outputs(outcome)
        docs = _retrieved_docs(outcome)
        verdict, issue_count = _review_summary(outcome)
        memories = _retrieved_memories(outcome)
        report.results.append(
            CaseResult(
                case_id=case.id,
                question=case.question,
                passed=passed,
                status=outcome.status,
                latency_ms=outcome.duration_ms,
                tokens=outcome.total_tokens,
                input_tokens=outcome.total_input_tokens,
                output_tokens=outcome.total_output_tokens,
                failed_checks=failed,
                hit_forbidden=forbidden,
                insufficient_data=insufficient,
                sql_rows=outputs.get("row_count"),
                sql_hash=outputs.get("sql_hash"),
                summary=(outcome.summary or "")[:200],
                error=outcome.error,
                route_tokens=outcome.tokens_by_route,
                retrieved_docs=sorted(set(docs)),
                knowledge_count=len(docs),
                review_verdict=verdict,
                review_issues=issue_count,
                memory_count=len(memories),
                memory_sunk=outcome.memory_sunk,
                memory_error=outcome.memory_error,
            )
        )
        if passed:
            report.passed += 1
        print(
            f"{'PASS' if passed else 'FAIL'} "
            f"{outcome.duration_ms / 1000:.1f}s {outcome.total_tokens}tok"
            + (f"  未命中={failed}" if failed else "")
            + (f"  禁用词={forbidden}" if forbidden else "")
        )

    report.wall_time_s = time.perf_counter() - started
    return report


# --------------------------------------------------------------------- 呈现


def render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"# 评测基线报告（{report.started_at}）")
    add("")
    add(f"- 模型：`{report.provider}/{report.model}`")
    add(f"- 用例数：{report.total_cases}，通过 **{report.passed}**，成功率 **{report.success_rate:.1%}**")
    add(f"- 墙钟总耗时：{report.wall_time_s / 60:.1f} 分钟")
    add("")

    add("## M1 时延与成功率")
    add("")
    add("| 指标 | 实测 | M1 目标 | 结论 |")
    add("| :--- | :--- | :--- | :--- |")
    add(f"| 成功率 | {report.success_rate:.1%} | ≥ 80% | {'✅ 达标' if report.success_rate >= 0.8 else '❌ 未达标'} |")
    add(f"| P50 端到端时延 | {report.p50_latency_s:.1f} 秒 | ≤ 60 秒 | {'✅' if report.p50_latency_s <= 60 else '❌'} |")
    add(f"| P95 端到端时延 | {report.p95_latency_s:.1f} 秒 | ≤ 60 秒 | {'✅' if report.p95_latency_s <= 60 else '❌'} |")
    add(f"| 最大时延 | {report.max_latency_s:.1f} 秒 | ≤ 60 秒 | {'✅' if report.max_latency_s <= 60 else '❌'} |")
    add("")

    add("## D2 单任务 Token 成本")
    add("")
    add("| 指标 | 实测 |")
    add("| :--- | :--- |")
    add(f"| 平均 Token/任务 | {report.mean_tokens:,.0f} |")
    add(f"| P95 Token/任务 | {report.p95_tokens:,} |")
    add(f"| 最大 Token/任务 | {report.max_tokens:,} |")
    suggestion = int(report.max_tokens * 1.5 / 10000 + 0.9999) * 10000
    add(f"| **建议 `DEFAULT_MAX_TOKENS`** | **{suggestion:,}**（最大实测 × 1.5 安全系数，向上取整到万） |")
    add("")

    # ---- 按调用点归因：分级路由的决策依据 ----
    routes = report.route_summary()
    if routes:
        llm_ms = sum(s["duration_ms"] for s in routes.values())
        wall_ms = sum(r.latency_ms for r in report.results)
        add("## 按调用点的成本/时延归因（分级路由依据）")
        add("")
        add(f"模型调用总耗时 {llm_ms / 1000:.1f} 秒，占全部用例墙钟耗时（{wall_ms / 1000:.1f} 秒）的 "
            f"**{100 * llm_ms / wall_ms if wall_ms else 0:.0f}%**——其余为数据库与编排开销。")
        add("")
        add("| 调用点 | 次数 | 总 token | 单次均值 | 总耗时 | 占模型调用耗时 | 单次均值 |")
        add("| :--- | --: | --: | --: | --: | --: | --: |")
        for route, stats in sorted(routes.items(), key=lambda kv: -kv[1]["duration_ms"]):
            calls = max(stats["calls"], 1)
            share = 100 * stats["duration_ms"] / llm_ms if llm_ms else 0
            add(
                f"| `{route}` | {stats['calls']} | {stats['total_tokens']:,} | "
                f"{stats['total_tokens'] // calls:,} | {stats['duration_ms'] / 1000:.1f}s | "
                f"{share:.0f}% | {stats['duration_ms'] / calls / 1000:.1f}s |"
            )
        add("")

    add("## 逐条结果")
    add("")
    add("| # | 用例 | 结果 | 时延 | Token | 知识 | 记忆 | SQL 行数 | 评审 | 未命中 |")
    add("| :-- | :--- | :-- | --: | --: | --: | --: | --: | :--- | :--- |")
    for i, r in enumerate(report.results, start=1):
        mark = "✅" if r.passed else "❌"
        detail = r.error or (", ".join(r.failed_checks) if r.failed_checks else "")
        if r.hit_forbidden:
            detail = f"命中禁用词 {r.hit_forbidden}"
        review = "-"
        if r.review_verdict:
            icon = {"pass": "✓", "pass_with_issues": "△"}.get(r.review_verdict, "✗")
            review = f"{icon}{r.review_verdict}"
        add(
            f"| {i} | `{r.case_id}` | {mark} | {r.latency_ms / 1000:.1f}s | "
            f"{r.tokens:,} | {r.knowledge_count or '-'} | {r.memory_count or '-'} | "
            f"{r.sql_rows if r.sql_rows is not None else '-'} | {review} | {detail} |"
        )
    add("")

    failures = [r for r in report.results if not r.passed]
    if failures:
        add("## 失败明细")
        add("")
        for r in failures:
            add(f"### `{r.case_id}`")
            add(f"- 问题：{r.question}")
            add(f"- 状态：{r.status}；未命中：{r.failed_checks}；禁用词：{r.hit_forbidden}")
            if r.error:
                add(f"- 错误：{r.error}")
            if r.summary:
                add(f"- 结论摘要：{r.summary}")
            add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m multiagent.eval.runner")
    parser.add_argument("--ids", help="只跑指定用例，逗号分隔")
    parser.add_argument("--limit", type=int, help="只跑前 N 条")
    parser.add_argument("--out", default="eval_results", help="结果输出目录")
    parser.add_argument(
        "--no-prereq",
        action="store_true",
        help="不自动补齐前置用例（用于记忆回归的「牙齿」演示：清空记忆后单跑消费方用例）",
    )
    args = parser.parse_args(argv)

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    cases = get_cases(ids=ids, limit=args.limit)
    if not cases:
        print("没有匹配的用例")
        return 1

    # 前置用例自动补齐并排到最前：记忆用例依赖产生记忆的分析用例，
    # 单跑消费方时必须先把前置用例跑掉，否则会以「记忆为空」的假象失败。
    if not args.no_prereq:
        selected_ids = {c.id for c in cases}
        prereqs: list[EvalCase] = []
        for case in cases:
            if case.requires_case and case.requires_case not in selected_ids:
                prereqs.extend(get_cases(ids=[case.requires_case]))
                selected_ids.add(case.requires_case)
        if prereqs:
            cases = prereqs + cases
            print(f"自动补齐前置用例：{[c.id for c in prereqs]}")

    settings = Settings()
    known = {c.id for c in CASES}
    unknown = [c.id for c in cases if c.id not in known]
    if unknown:
        print(f"忽略未知用例 id: {unknown}")

    print(f"▶ 评测 {len(cases)} 条用例，模型 {settings.primary.provider}/{settings.primary.model}\n")

    # RAG 用例依赖知识库；先确认它非空，否则这些用例会以难懂的方式失败
    knowledge_cases = [c for c in cases if c.expect_docs or c.require_knowledge]
    if knowledge_cases:
        from ..rag import KnowledgeStore

        try:
            stats = KnowledgeStore(settings).stats()
        except Exception as exc:
            print(f"⚠ 无法确认知识库状态（{exc}）——本轮 {len(knowledge_cases)} 条 RAG 用例可能全部失败\n")
        else:
            if stats["chunks"] == 0:
                print(
                    f"⚠ 知识库为空，但本轮有 {len(knowledge_cases)} 条用例依赖它。"
                    "请先执行：python -m multiagent.rag.ingest\n"
                )
            else:
                print(
                    f"知识库就绪：{stats['documents']} 篇 / {stats['chunks']} 分块"
                    f"（{len(knowledge_cases)} 条用例依赖它）\n"
                )

    # 记忆用例依赖记忆库里有内容（由前置用例沉淀）；打印现状便于识别「假失败」
    if any(c.expect_memories or c.requires_case for c in cases):
        from ..memory import MemoryStore

        try:
            mem_stats = MemoryStore(settings).stats()
            print(
                f"长期记忆：{mem_stats['total']} 条（validated {mem_stats['validated']}）"
                "——记忆用例依赖前置用例先沉淀\n"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ 无法读取长期记忆状态（{exc}）\n")

    report = run_cases_sync(settings, cases)

    markdown = render_markdown(report)
    print("\n" + markdown)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(TZ_CN).strftime("%Y%m%d-%H%M%S")
    (out_dir / f"baseline-{stamp}.md").write_text(markdown, encoding="utf-8")
    (out_dir / f"baseline-{stamp}.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {out_dir}/baseline-{stamp}.md / .json")
    return 0 if report.success_rate >= 0.8 else 1


def run_cases_sync(settings: Settings, cases: list[EvalCase]) -> EvalReport:
    return run_async(run_cases(settings, cases))


if __name__ == "__main__":
    raise SystemExit(main())
