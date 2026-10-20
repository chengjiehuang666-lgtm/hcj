"""命令行入口：把一句自然语言需求跑成分析报告。

用法：
    python -m multiagent.cli "分析上月华东区销售额下滑的原因"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .asyncutil import run_async
from .config import Settings
from .graph import AnalysisPipeline
from .graph.pipeline import new_task_event
from .observability import configure_logging
from .security.dlp import redact_structure, redact_text

DEFAULT_QUESTION = "分析上月华东区销售额下滑的原因"


def _redacted_outcome(outcome):
    """CLI 是用户可见边界：打印前统一做 DLP 脱敏（内部状态不动）。"""
    return replace(
        outcome,
        summary=redact_text(outcome.summary or "")[0],
        error=redact_text(outcome.error)[0] if outcome.error else None,
        node_results=redact_structure(outcome.node_results),
        node_outputs=redact_structure(outcome.node_outputs),
        blackboard=redact_structure(outcome.blackboard),
        dag=redact_structure(outcome.dag),
        approval_requests=redact_structure(outcome.approval_requests),
        approvals=redact_structure(outcome.approvals),
    )


def _print_outcome(outcome) -> int:
    print("=" * 72)
    print(f"任务 ID   : {outcome.task_id}")
    print(f"状态      : {outcome.status}")
    print(f"耗时      : {outcome.duration_ms / 1000:.1f} 秒")
    print(f"Token     : {outcome.total_tokens}（输入 {outcome.total_input_tokens} / 输出 {outcome.total_output_tokens}）")
    print(f"成本      : {outcome.total_cost:.4f}（按调用点/Agent 明细见 /metrics）")
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

    analyst_results = [r for r in outcome.node_results if r.get("agent_role") == "analyst"]
    if analyst_results:
        outputs = outcome.merged_outputs_for_role("analyst")
        print("\n【关键发现】")
        for finding in outputs.get("key_findings", []):
            print(f"  · {finding}")
        print(f"\n【执行过的 SQL（哈希 {outputs.get('sql_hash')}）】")
        print(f"  {outputs.get('sql')}")
        print(f"  返回 {outputs.get('row_count')} 行")
        if outputs.get("insufficient_data"):
            print("\n  ⚠ 模型判定数据不足以支撑结论")

    # 文档场景（第 2 场景）：产物本体是文档
    if any(r.get("agent_role") == "writer" for r in outcome.node_results):
        document = outcome.merged_outputs_for_role("writer")
        print(f"\n【文档】{document.get('title', '(无标题)')}")
        print(document.get("document_markdown", ""))
        for gap in document.get("material_gaps", []) or []:
            print(f"  ⚠ 素材未涵盖：{gap}")

    # 评审结论（reviewer 节点存在时）
    if any(r.get("agent_role") == "reviewer" for r in outcome.node_results):
        review = outcome.merged_outputs_for_role("reviewer")
        print(
            f"\n【评审】verdict = {review.get('verdict')}"
            f"（问题 {review.get('issue_count', 0)} 个）"
        )
        for check in review.get("checks", []) or []:
            print(f"  · [{check.get('result')}] {str(check.get('criterion'))[:60]}")
        for issue in review.get("issues", []) or []:
            print(f"  ⚠ [{issue.get('severity')}] {str(issue.get('description'))[:100]}")

    print("\n【黑板条目】")
    for key, entry in outcome.blackboard.items():
        by = entry.get("written_by") or {}
        who = f"{by.get('agent_role')}@{by.get('agent_version')}({by.get('node_id')})"
        print(f"  · {key} v{entry.get('version')} [{entry.get('status')}] by {who} "
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
    configure_logging(settings)
    event = new_task_event(args.question, settings=settings)
    print(f"▶ 提交需求：{redact_text(args.question)[0]}")
    print(f"▶ 模型：{settings.primary.provider}/{settings.primary.model} "
          f"（备用 {settings.fallback.provider}/{settings.fallback.model}）\n")

    outcome = run_async(AnalysisPipeline(settings).run(event))
    if settings.security_dlp_enabled:
        outcome = _redacted_outcome(outcome)
    return _print_outcome(outcome)


if __name__ == "__main__":
    sys.exit(main())
