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


def _analyst_outputs(outcome) -> dict:
    for res in outcome.node_results:
        if res.get("node_id") == "analyst_1":
            return res.get("outputs", {}) or {}
    return {}


def grade(case: EvalCase, outcome) -> tuple[bool, list[str], list[str], bool]:
    """返回 (是否通过, 未命中的检查项, 命中的禁用词, 是否判定数据不足)。"""
    outputs = _analyst_outputs(outcome)
    insufficient = bool(outputs.get("insufficient_data"))

    text = normalize(
        " ".join(
            [
                outcome.summary or "",
                " ".join(outputs.get("key_findings", []) or []),
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
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            print("崩溃")
            continue

        passed, failed, forbidden, insufficient = grade(case, outcome)
        outputs = _analyst_outputs(outcome)
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

    add("## 逐条结果")
    add("")
    add("| # | 用例 | 结果 | 时延 | Token | SQL 行数 | 未命中 |")
    add("| :-- | :--- | :-- | --: | --: | --: | :--- |")
    for i, r in enumerate(report.results, start=1):
        mark = "✅" if r.passed else "❌"
        detail = r.error or (", ".join(r.failed_checks) if r.failed_checks else "")
        if r.hit_forbidden:
            detail = f"命中禁用词 {r.hit_forbidden}"
        add(
            f"| {i} | `{r.case_id}` | {mark} | {r.latency_ms / 1000:.1f}s | "
            f"{r.tokens:,} | {r.sql_rows if r.sql_rows is not None else '-'} | {detail} |"
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
    args = parser.parse_args(argv)

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    cases = get_cases(ids=ids, limit=args.limit)
    if not cases:
        print("没有匹配的用例")
        return 1

    settings = Settings()
    known = {c.id for c in CASES}
    unknown = [c.id for c in cases if c.id not in known]
    if unknown:
        print(f"忽略未知用例 id: {unknown}")

    print(f"▶ 评测 {len(cases)} 条用例，模型 {settings.primary.provider}/{settings.primary.model}\n")
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
