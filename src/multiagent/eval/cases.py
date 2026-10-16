"""评测用例集。

用于校准《实施路线图》3.6 的待定项 D2（单任务 Token 上限）与
M1 的时延/成功率基线（「端到端时延 ≤ 60 秒、成功率 ≥ 80%」）。

用例基于 `docker/initdb/02-demo-data.sql` 的演示数据构造，因此**期望值是可独立
验算的**——不是让模型自己判断对错，而是拿已知事实去核对它的结论。

判定方式见 `runner.grade()`：
- `expect_any` 是若干**备选组**，每组至少命中一个字符串才算通过；
- `forbid` 中的字符串一个都不许出现（用于抓编造）。

数字统一去除逗号与空格后比对，因此「2,335,000」与「2335000」等价。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expect_any: list[list[str]] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    expect_insufficient: bool = False
    note: str = ""


# 演示数据的事实（**由数据库实测算出**，改动 02-demo-data.sql 时须重算）：
#   华东 07=2,850,000 orders=11,000 refund=125,000 率 4.39%
#   华东 08=2,950,000 orders=11,400 refund=130,000 率 4.41%
#   华东 09=2,335,000 orders= 8,830 refund=234,500 率 10.04%  → 环比 -20.85%
#   华南 07=1,760,000  08=1,805,000  09=1,794,000            → 环比 -0.61%
#   华东 线上 08=1,670,000 09=1,080,000；门店 08=1,280,000 09=1,255,000（-1.95%）
#   华东 数码 08=2,170,000 09=1,680,000；线上数码 08=1,250,000 09=780,000（-37.60%）
#   华东 家居 08=  780,000 09=  655,000
#   华东 退货额 09=234,500；全区域 09 合计=4,129,000；华东占比 56.55%
#   华东-华南差额 09 = 541,000

CASES: list[EvalCase] = [
    EvalCase(
        id="core_decline_cause",
        question="分析上月华东区销售额下滑的原因",
        expect_any=[["2335000"], ["2950000"], ["20.8", "20.9"], ["线上", "数码"]],
        note=(
            "核心问题。期望值对齐 Planner 的验收标准（总额、环比幅度、贡献最大的一个维度），"
            "**不要求**下钻到「线上×数码」交叉格子——那超出规格。"
            "交叉格子的下钻由 east_online_digital 单独覆盖。"
        ),
    ),
    EvalCase(
        id="east_sep_revenue",
        question="华东区 2026 年 9 月的销售额是多少？",
        expect_any=[["2335000"]],
    ),
    EvalCase(
        id="east_vs_south",
        question="对比华东区和华南区 9 月的销售表现，哪个下滑更严重？",
        expect_any=[["2335000"], ["1794000"], ["华东"]],
    ),
    EvalCase(
        id="east_channel_breakdown",
        question="华东区 9 月线上渠道和门店渠道的销售额分别是多少？",
        expect_any=[["1080000"], ["1255000"]],
    ),
    EvalCase(
        id="east_online_digital",
        question="华东区线上数码品类 9 月销售额相比 8 月下降了多少？",
        expect_any=[["780000"], ["1250000"], ["37.6"]],
    ),
    EvalCase(
        id="east_refund_rate",
        question="华东区 9 月的退款率相比 8 月有什么变化？",
        # 区域整体口径为 10.04% / 4.41%；若模型下钻到品类则为 10.54% / 4.19%。
        # 另需接受**小数形式**（0.1004 / 0.0441），模型可能不以百分比表述。
        expect_any=[
            ["10.0", "10.5", "0.100", "0.104", "0.105"],
            ["4.1", "4.4", "0.041", "0.044"],
        ],
        note="口径易混（区域整体 vs 品类分项），且百分比/小数两种表述都应接受",
    ),
    EvalCase(
        id="east_orders_trend",
        question="华东区 8 月和 9 月的订单量分别是多少，变化如何？",
        expect_any=[["11400"], ["8830"]],
    ),
    EvalCase(
        id="east_category_digital",
        question="华东区数码品类的销售额 9 月比 8 月减少了多少？",
        expect_any=[["1680000"], ["2170000"]],
    ),
    EvalCase(
        id="east_store_flat",
        question="华东区门店渠道 9 月是否也出现了明显下滑？",
        # 1.953% 四舍五入为 2.0%，模型写 2.0 也正确；「并未明显下滑」与「持平」等价。
        # 打分器曾因只接受 "1.9"/"持平" 而把正确答案判失败——属于打分器缺陷，不是模型问题。
        expect_any=[
            ["1255000"],
            ["1.9", "2.0", "持平", "未出现明显下滑", "没有出现明显下滑", "基本持平"],
        ],
    ),
    EvalCase(
        id="all_regions_sep_total",
        question="所有区域 9 月的销售额合计是多少？",
        expect_any=[["4129000"]],
    ),
    EvalCase(
        id="three_month_trend",
        question="华东区 7 月、8 月、9 月的销售额趋势如何？",
        expect_any=[["2850000"], ["2950000"], ["2335000"]],
    ),
    EvalCase(
        id="peak_month",
        question="华东区这三个月里哪个月销售额最高？",
        expect_any=[["2950000"]],
    ),
    EvalCase(
        id="south_stable",
        question="华南区 9 月的销售表现是否稳定？",
        # 问题问的是「是否稳定」，因此要求：稳定性判断 + 一个量化幅度。
        # 不强制给出总额——那是过度指定（曾因此把正确答案判失败）。
        expect_any=[["稳定", "持平"], ["0.3", "0.6", "0.8", "小幅", "1%"]],
        note="稳定性判断 + 量化幅度；不要求列出总额",
    ),
    EvalCase(
        id="biggest_contributor",
        question="华东区 9 月销售额下滑中，哪个品类或渠道贡献最大？",
        expect_any=[["数码", "线上"]],
        note="品类口径（数码）与渠道口径（线上）都是正确答案",
    ),
    EvalCase(
        id="east_home_category",
        question="华东区家居品类 9 月销售额是多少？",
        expect_any=[["655000"]],
    ),
    EvalCase(
        id="refund_amount_east",
        question="华东区 9 月的退货金额合计是多少？",
        expect_any=[["234500"]],
    ),
    EvalCase(
        id="east_share_of_total",
        question="华东区销售额占全部区域的比例在 9 月大约是多少？",
        expect_any=[["56.5", "56.6"]],
    ),
    EvalCase(
        id="two_region_sep_gap",
        question="9 月华东区比华南区多多少销售额？",
        expect_any=[["541000"]],
    ),
    EvalCase(
        id="metric_definition_used",
        question="你们统计销售额时用的是什么口径？",
        expect_any=[["支付", "不含", "完成", "退款"], ["财务", "口径"]],
        note="口语化提问：应先取口径再回答，考察 Retriever 的作用",
    ),
    EvalCase(
        id="missing_data_honesty",
        question="华北区 9 月的销售额下滑了多少？",
        expect_any=[["没有", "无数据", "缺失", "不存在", "查不到", "未找到", "未覆盖", "0 行"]],
        expect_insufficient=True,
        note="关键诚实性用例：数据里根本没有华北区，必须承认而不是编造",
    ),
]


def get_cases(ids: list[str] | None = None, limit: int | None = None) -> list[EvalCase]:
    selected = CASES
    if ids:
        wanted = set(ids)
        selected = [c for c in CASES if c.id in wanted]
    if limit is not None:
        selected = selected[:limit]
    return list(selected)
