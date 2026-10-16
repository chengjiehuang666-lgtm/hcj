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
    ) -> None:
        self.status = status
        self.summary = summary
        self.node_results = [
            {
                "node_id": "analyst_1",
                "outputs": {
                    "key_findings": findings or [],
                    "insufficient_data": insufficient,
                    "row_count": row_count,
                },
            }
        ]


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
