"""评测层：评测集、执行器与基线报告。

用途与产出见《实施路线图》3.2（W4 交付物：评测集与成功率/时延基线报告）
与 3.6（待定项 D2：单任务 Token 成本上限）。
"""

from .cases import CASES, EvalCase, get_cases
from .runner import EvalReport, CaseResult, grade, normalize, run_cases, run_cases_sync

__all__ = [
    "CASES",
    "EvalCase",
    "get_cases",
    "EvalReport",
    "CaseResult",
    "grade",
    "normalize",
    "run_cases",
    "run_cases_sync",
]
