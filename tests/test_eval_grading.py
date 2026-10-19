"""评测打分逻辑测试。

打分器如果本身不准，基线数据就是假的——所以它必须先被测试。
这里用构造的 outcome 注入，不调用模型。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiagent.eval.cases import EvalCase  # noqa: E402
from multiagent.eval.runner import _alternatives, grade, normalize  # noqa: E402


class FakeOutcome:
    def __init__(
        self,
        *,
        status: str = "ok",
        summary: str = "",
        findings: list[str] | None = None,
        insufficient: bool = False,
        row_count: int | None = 3,
        knowledge: list[dict] | None = None,
        memories: list[dict] | None = None,
        document: str | None = None,
    ) -> None:
        self.status = status
        self.summary = summary
        # agent_role 必须带上：打分器按 role 取值（动态 DAG 下 node_id 可变）
        self.node_results = [
            {
                "node_id": "retriever_1",
                "agent_role": "retriever",
                "outputs": {"knowledge": knowledge or [], "memories": memories or []},
            },
            {
                "node_id": "analyst_1",
                "agent_role": "analyst",
                "outputs": {
                    "key_findings": findings or [],
                    "insufficient_data": insufficient,
                    "row_count": row_count,
                },
            },
        ]
        if document is not None:
            self.node_results.append(
                {
                    "node_id": "writer_1",
                    "agent_role": "writer",
                    "status": "ok",
                    "outputs": {"document_markdown": document},
                }
            )


# ------------------------------------------------------------------ 归一化


def test_normalize_strips_commas_and_space() -> None:
    assert normalize("2,335,000 元") == "2335000元"
    assert normalize("1 080 000") == "1080000"


def test_alternatives_adds_wan_notation_for_large_numbers() -> None:
    alts = _alternatives("2950000")
    assert "2950000" in alts
    assert "295万" in alts, alts


def test_alternatives_leaves_non_numeric_untouched() -> None:
    assert _alternatives("20.8") == ["20.8"]
    assert _alternatives("线上") == ["线上"]


def test_alternatives_ignores_small_numbers() -> None:
    assert _alternatives("8830") == ["8830"]


# -------------------------------------------------------------------- 打分


def test_grade_passes_when_all_groups_matched() -> None:
    case = EvalCase(id="c", question="q", expect_any=[["2335000"], ["20.8"]])
    outcome = FakeOutcome(summary="华东区 9 月销售额 2,335,000 元，环比下降 20.85%。")
    passed, failed, forbidden, _ = grade(case, outcome)
    assert passed, failed
    assert not failed and not forbidden


def test_grade_accepts_wan_notation() -> None:
    case = EvalCase(id="c", question="q", expect_any=[["2950000"]])
    outcome = FakeOutcome(summary="峰值出现在 8 月，销售额约 295万元。")
    passed, failed, _, _ = grade(case, outcome)
    assert passed, failed


def test_grade_fails_on_missing_group() -> None:
    case = EvalCase(id="c", question="q", expect_any=[["2335000"], ["37.6"]])
    outcome = FakeOutcome(summary="华东区 9 月销售额 2335000 元。")
    passed, failed, _, _ = grade(case, outcome)
    assert not passed
    assert failed == ["37.6"], failed


def test_grade_detects_forbidden_token() -> None:
    case = EvalCase(id="c", question="q", forbid=["9999999"])
    outcome = FakeOutcome(summary="结论是 9999999 元。")
    passed, _, forbidden, _ = grade(case, outcome)
    assert not passed
    assert forbidden == ["9999999"]


def test_grade_uses_findings_too_not_only_summary() -> None:
    case = EvalCase(id="c", question="q", expect_any=[["1080000"]])
    outcome = FakeOutcome(summary="整体下滑。", findings=["线上渠道为 1,080,000 元。"])
    passed, failed, _, _ = grade(case, outcome)
    assert passed, failed


def test_grade_respects_expect_insufficient_as_pass_path() -> None:
    """模型显式声明数据不足时，视作满足诚实性用例，不强制命中字面词。"""
    case = EvalCase(
        id="missing", question="q", expect_any=[["无数据"]], expect_insufficient=True
    )
    outcome = FakeOutcome(summary="该区域无相关记录。", insufficient=True)
    passed, failed, _, insufficient = grade(case, outcome)
    assert passed, failed
    assert insufficient is True


def test_grade_fails_when_pipeline_failed() -> None:
    case = EvalCase(id="c", question="q", expect_any=[["1"]])
    outcome = FakeOutcome(status="failed")
    passed, failed, _, _ = grade(case, outcome)
    assert not passed
    assert failed == ["pipeline_failed"]


def test_case_set_is_well_formed() -> None:
    """用例集自身的完整性：id 唯一、问题非空、20 条基线。"""
    from multiagent.eval.cases import CASES

    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "用例 id 必须唯一"
    assert all(c.question.strip() for c in CASES)
    assert len(CASES) >= 20, f"基线要求至少 20 条，当前 {len(CASES)}"
    assert all(c.expect_any or c.forbid or c.expect_insufficient for c in CASES), (
        "每条用例都要有可判定的期望，否则无法自动打分"
    )


# ------------------------------------------------- 知识库依赖用例（RAG 回归）


def test_rag_cases_exist_and_are_structural() -> None:
    """RAG 能力必须有回归用例，否则会在后续改动中被无声改坏。"""
    from multiagent.eval.cases import CASES

    rag_cases = [c for c in CASES if c.expect_docs or c.require_knowledge]
    assert len(rag_cases) >= 4, f"知识库依赖用例应至少 4 条，当前 {len(rag_cases)}"
    assert any("refund" in c.id for c in rag_cases), "应包含退货率那条核心用例"
    # 每条 RAG 用例都必须同时检查「检索到了什么」与「答案说了什么」
    assert all(c.expect_any and c.expect_docs for c in rag_cases), (
        "RAG 用例既要检查检索结果，也要检查答案内容"
    )


def test_expected_doc_retrieved_passes() -> None:
    case = EvalCase(id="c", question="q", expect_docs=["return-policy"])
    outcome = FakeOutcome(knowledge=[{"doc_id": "return-policy-2026h2"}])
    passed, failed, _, _ = grade(case, outcome)
    assert passed, failed


def test_expected_doc_missing_fails_even_if_answer_looks_right() -> None:
    """答案看起来对，但知识库没被检索到 —— 必须判失败。

    这正是「只检查答案文本无法区分模型本来就知道与真的查了知识库」的落点。
    """
    case = EvalCase(id="c", question="q", expect_docs=["return-policy"])
    outcome = FakeOutcome(summary="这是政策口径变化导致的", knowledge=[{"doc_id": "other-doc"}])
    passed, failed, _, _ = grade(case, outcome)
    assert not passed
    assert any("知识库" in f for f in failed), failed


def test_require_knowledge_fails_when_nothing_retrieved() -> None:
    case = EvalCase(id="c", question="q", require_knowledge=True)
    passed, failed, _, _ = grade(case, FakeOutcome(knowledge=[]))
    assert not passed
    assert "未检索到任何知识" in failed


def test_doc_matching_is_substring_based() -> None:
    case = EvalCase(id="c", question="q", expect_docs=["channel-ops"])
    outcome = FakeOutcome(knowledge=[{"doc_id": "channel-ops-calendar"}])
    passed, failed, _, _ = grade(case, outcome)
    assert passed, failed


def test_multiple_expected_docs_accept_any_match() -> None:
    case = EvalCase(id="c", question="q", expect_docs=["sales-data-dictionary", "region-definition"])
    outcome = FakeOutcome(knowledge=[{"doc_id": "region-definition"}])
    passed, failed, _, _ = grade(case, outcome)
    assert passed, failed


# --------------------------------------------------------- 记忆用例门槛（A）


def test_grade_memory_expectation_is_retrieval_structural() -> None:
    """grade 的记忆断言只查「召回里是否有预期内容」（结构性证据）。"""
    case = EvalCase(id="mem", question="q", expect_memories=["华东"])
    hit = FakeOutcome(memories=[{"content": "华东区下滑", "source_ref": "任务 x"}])
    assert grade(case, hit)[0] is True
    miss = FakeOutcome(memories=[{"content": "华南区稳定", "source_ref": "任务 y"}])
    passed, failed, _, _ = grade(case, miss)
    assert passed is False and any("记忆:华东" in f for f in failed)


def test_memory_source_problem_requires_producer_sink() -> None:
    """前置用例的沉淀结果必须真的落库：断言查前置用例的 memory_sunk。

    这条门槛不能绑成「召回里含前置用例的 task_id」——那取决于检索排名，
    相似历史记忆一多就不保证进 top-k（W6 修复首版因此误判，已改正）。
    """
    from multiagent.eval.runner import CaseResult, _memory_source_problem

    case = EvalCase(
        id="mem", question="q", expect_memories=["华东"], memory_source_case="producer"
    )

    def result(case_id: str, sunk: int | None) -> CaseResult:
        return CaseResult(
            case_id=case_id,
            question="q",
            passed=True,
            status="ok",
            latency_ms=1,
            tokens=1,
            input_tokens=1,
            output_tokens=0,
            memory_sunk=sunk,
        )

    assert _memory_source_problem(case, [result("producer", 42)]) is None
    assert (
        _memory_source_problem(case, [result("producer", None)])
        == "前置用例未沉淀:producer"
    )
    assert (
        _memory_source_problem(case, [])
        == "前置用例未运行:producer"
    )


def test_memory_source_problem_noop_without_binding() -> None:
    from multiagent.eval.runner import _memory_source_problem

    plain = EvalCase(id="mem", question="q", expect_memories=["华东"])
    assert _memory_source_problem(plain, []) is None


# ------------------------------------------------- W8：50 条规模与场景约束


def test_case_set_meets_w8_scale_and_scenarios() -> None:
    """W8：50 条，每场景 ≥20（文档 20 / 非文档 30）；结构性字段必须健全。"""
    from multiagent.eval.cases import CASES
    from multiagent.rag.corpus import DOCUMENTS

    assert len(CASES) >= 50, f"W8 要求 50 条，当前 {len(CASES)}"
    doc = [c for c in CASES if c.scenario == "document"]
    non_doc = [c for c in CASES if c.scenario != "document"]
    assert len(doc) >= 20, f"文档场景应 ≥20 条，当前 {len(doc)}"
    assert len(non_doc) >= 30, f"非文档场景应 ≥30 条，当前 {len(non_doc)}"

    for case in doc:
        assert {"writer", "reviewer"} <= set(case.expect_roles), case.id
        assert case.expect_docs and case.expect_any, case.id
        assert case.require_knowledge, case.id

    # expect_docs 必须能在语料里找到（拼错只会在运行期失败，这里提前拦住）
    for case in CASES:
        for exp in case.expect_docs:
            assert any(exp in d.doc_id for d in DOCUMENTS), (
                f"{case.id}: expect_docs {exp!r} 不在语料中"
            )

    # requires_case 必须指向存在的用例
    ids = {c.id for c in CASES}
    for case in CASES:
        if case.requires_case:
            assert case.requires_case in ids, (
                f"{case.id}: requires_case {case.requires_case!r} 不存在"
            )


def test_document_markdown_is_graded() -> None:
    """文档用例的断言必须落在 document_markdown 上（不能只靠 summary）。"""
    case = EvalCase(
        id="doc_x",
        question="q",
        scenario="document",
        expect_any=[["96%"]],
        expect_docs=["q2-2026-review"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
    )
    from types import SimpleNamespace

    def outcome(document: str) -> FakeOutcome:
        out = FakeOutcome(
            document=document,
            knowledge=[{"doc_id": "q2-2026-review"}],
        )
        out.node_results.append(
            {"node_id": "reviewer_1", "agent_role": "reviewer", "status": "ok",
             "outputs": {"verdict": "pass"}}
        )
        return out

    assert grade(case, outcome("# 简报\n\nQ2 达成率 96%"))[0] is True
    passed, failed, _, _ = grade(case, outcome("# 简报\n\n业绩良好"))
    assert passed is False and any("96%" in f for f in failed)


def test_sum_route_tokens_merges_segments() -> None:
    """暂停/恢复是两段 run()：按调用点归因必须相加，否则报告低估成本。"""
    from types import SimpleNamespace

    from multiagent.eval.runner import _sum_route_tokens

    seg1 = SimpleNamespace(
        tokens_by_route={"planner": {"calls": 1, "total_tokens": 100}},
        duration_ms=100,
    )
    seg2 = SimpleNamespace(
        tokens_by_route={
            "writer": {"calls": 1, "total_tokens": 300},
            "planner": {"calls": 1, "total_tokens": 50},
        },
        duration_ms=200,
    )
    merged = _sum_route_tokens([seg1, seg2])
    assert merged["planner"]["calls"] == 2
    assert merged["planner"]["total_tokens"] == 150
    assert merged["writer"]["total_tokens"] == 300


def test_run_cases_auto_approves_multi_gate_and_merges_latency(monkeypatch) -> None:
    """多个审批门：自动批准循环（≤3 次），且时延按分段相加。"""
    import asyncio
    from types import SimpleNamespace

    from multiagent.eval import runner as runner_module
    from multiagent.graph.pipeline import PipelineOutcome

    class FakePipeline:
        def __init__(self, settings):
            self.calls = 0

        async def run(self, event, *, resume=False, resume_value=None):
            self.calls += 1
            if self.calls <= 2:
                return PipelineOutcome(
                    task_id="t",
                    status="awaiting_approval",
                    duration_ms=100,
                    approval_requests=[{"id": "i"}],
                )
            return PipelineOutcome(
                task_id="t",
                status="ok",
                duration_ms=200,
                total_tokens=10,
                node_results=[
                    {
                        "agent_role": "writer",
                        "status": "ok",
                        "outputs": {"document_markdown": "达成率 96%"},
                    }
                ],
                summary="",
            )

    monkeypatch.setattr(runner_module, "AnalysisPipeline", FakePipeline)
    case = EvalCase(
        id="doc_multi_gate",
        question="q",
        scenario="document",
        expect_any=[["96%"]],
    )
    from multiagent.config import Settings

    report = asyncio.run(runner_module.run_cases(Settings(_env_file=None), [case]))

    result = report.results[0]
    assert result.passed is True
    assert result.latency_ms == 400, "三段 run() 的时延必须相加"
    assert result.status == "ok"
