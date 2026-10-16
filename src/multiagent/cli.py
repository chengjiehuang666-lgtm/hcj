"""命令行入口：把一句自然语言需求跑成分析报告。

用法：
    python -m multiagent.cli "分析上月华东区销售额下滑的原因"
"""

from __future__ import annotations

import argparse
import sys

from .asyncutil import run_async
from .config import Settings
from .graph import AnalysisPipeline
from .graph.pipeline import new_task_event

DEFAULT_QUESTION = "分析上月华东区销售额下滑的原因"


def _print_outcome(outcome) -> int:
    print("=" * 72)
    print(f"任务 ID   : {outcome.task_id}")
    print(f"状态      : {outcome.status}")
    print(f"耗时      : {outcome.duration_ms / 1000:.1f} 秒")
    print(f"Token     : {outcome.total_tokens}（输入 {outcome.total_input_tokens} / 输出 {outcome.total_output_tokens}）")
    print("=" * 72)

    if outcome.status != "ok":
        print(f"\n✗ 执行失败：{outcome.error}")
        return 1

    # 规划结果
    dag = outcome.dag or {}
    print(f"\n【规划】目标：{dag.get('goal', '(未提供)')}")
    for node in dag.get("nodes", []):
        deps = node.get("depends_on") or []
        arrow = f" ← {', '.join(deps)}" if deps else ""
        print(f"  · {node['node_id']} [{node['agent_role']}]{arrow}")
        for criterion in node.get("acceptance", []):
            print(f"      验收：{criterion}")

    # 各节点执行情况
    print("\n【执行】")
    for res in outcome.node_results:
        tokens = res.get("tokens_used", {}).get("total", 0)
        conf = res.get("confidence")
        conf_text = f" 置信度 {conf:.2f}" if conf is not None else ""
        print(
            f"  · {res['node_id']:14s} 状态={res['status']:6s} "
            f"耗时={res.get('duration_ms', 0):>5d}ms token={tokens}{conf_text}"
        )

    # 可回溯性
    print("\n【溯源】")
    for res in outcome.node_results:
        for src in res.get("sources", []):
            print(f"  · {res['node_id']}: {src.get('type')} -> {src.get('ref')}")

    # 结论
    print("\n【结论】")
    print(f"  {outcome.summary or '(无)'}")

    analyst = next(
        (r for r in outcome.node_results if r.get("node_id") == "analyst_1"), None
    )
    if analyst:
        outputs = analyst.get("outputs", {})
        print("\n【关键发现】")
        for finding in outputs.get("key_findings", []):
            print(f"  · {finding}")
        print(f"\n【执行过的 SQL（哈希 {outputs.get('sql_hash')}）】")
        print(f"  {outputs.get('sql')}")
        print(f"  返回 {outputs.get('row_count')} 行")
        if outputs.get("insufficient_data"):
            print("\n  ⚠ 模型判定数据不足以支撑结论")

    print("\n【黑板条目】")
    for key, entry in outcome.blackboard.items():
        print(f"  · {key} v{entry.get('version')} by {entry.get('written_by')} "
              f"(sources={len(entry.get('sources', []))})")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m multiagent.cli",
        description="多 Agent 数据分析流水线（阶段一 MVP）",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f"自然语言分析需求（默认：{DEFAULT_QUESTION}）",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    event = new_task_event(args.question)
    print(f"▶ 提交需求：{args.question}")
    print(f"▶ 模型：{settings.primary.provider}/{settings.primary.model} "
          f"（备用 {settings.fallback.provider}/{settings.fallback.model}）\n")

    outcome = run_async(AnalysisPipeline(settings).run(event))
    return _print_outcome(outcome)


if __name__ == "__main__":
    sys.exit(main())
