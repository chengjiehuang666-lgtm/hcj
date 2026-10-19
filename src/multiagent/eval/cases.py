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
    #: 场景归属（W8 起用于「每场景 ≥20」的约束与分场景统计）：
    #: analysis | document | memory
    scenario: str = "analysis"
    expect_any: list[list[str]] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    expect_insufficient: bool = False
    #: 期望被检索到的知识库文档（doc_id 子串，命中**任意一个**即可）。
    #: 这是「知识库确实参与了这个问题的回答」的**结构性证据**——
    #: 只检查答案文本无法区分「模型本来就知道」与「真的查了知识库」。
    expect_docs: list[str] = field(default_factory=list)
    #: 是否强制要求检索阶段取到了知识（用于纯知识类问题）
    require_knowledge: bool = False
    #: 必须出现且 status=ok 的节点角色（结构性证据：该能力真的参与了执行）。
    #: 例如文档场景要求 ["writer", "reviewer"]；分析场景要求 ["reviewer"]。
    expect_roles: list[str] = field(default_factory=list)
    #: 期望被检索到的长期记忆内容（子串，命中**任意一个**即可）。
    #: 同样只查检索结果这一结构性证据，不看答案措辞。
    expect_memories: list[str] = field(default_factory=list)
    #: 前置用例：本用例依赖它先执行（如记忆用例依赖产生记忆的分析用例）。
    #: runner 会自动补齐并把前置用例排到前面；`--no-prereq` 可关闭（用于牙齿演示）。
    requires_case: str | None = None
    #: 记忆前置条件：要求该前置用例**本次运行真的沉淀**（memory_sunk 非空）。
    #: 只查「召回命中含华东」不够——开发库里的历史沉淀会让断言假通过（独立复核）；
    #: 但也不能要求「召回里必须有本次那条」——那取决于检索排名，不是产品承诺。
    memory_source_case: str | None = None
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
    # ---------------------- 知识库依赖用例（RAG 回归） ----------------------
    # 共同设计原则：**SQL 单独答不出来**。
    # 若哪天 RAG 被改坏（检索退化、知识没进提示词、灌库丢失），这些用例会直接失败——
    # 这是它们存在的唯一理由，也是「没有回归用例的能力等于没有能力」的落实。
    EvalCase(
        id="rag_refund_is_policy_not_quality",
        question="华东区9月线上退货率大幅上升，是不是产品质量出了问题？",
        expect_any=[
            # 只有读了政策文档才写得出这些词
            ["政策", "口径", "全量质检", "免运费"],
            # 关键结论：不能据此判定质量恶化
            ["非质量", "不代表", "并非", "不是质量", "不能说明", "不是产品", "口径变化", "政策调整"],
        ],
        expect_docs=["return-policy"],
        note="核心 RAG 用例：数据只显示退货率上升，能否解释成「口径变化而非质量恶化」完全依赖知识库",
    ),
    EvalCase(
        id="rag_region_definition",
        question="华东区具体包含哪些省市？",
        expect_any=[["江苏"], ["浙江"], ["山东", "安徽", "福建", "江西"]],
        expect_docs=["region-definition"],
        require_knowledge=True,
        note="纯知识问题：SQL 里只有「华东」两个字，省市清单只存在于知识库",
    ),
    EvalCase(
        id="rag_daily_volatility_infeasible",
        question="帮我算一下华东区9月的日波动率",
        expect_any=[
            ["月度", "月初", "快照", "月粒度"],
            ["无法", "不能", "做不到", "缺少", "没有日", "不足以"],
        ],
        expect_docs=["sales-data-dictionary", "region-definition"],
        note="可行性判断依赖数据字典：数据是月度快照，日粒度指标算不出来",
    ),
    EvalCase(
        id="rag_sep_promo_anomaly",
        question="9 月是线上大促月，华东区线上数码 9 月却不升反降，这说明了什么？",
        expect_any=[
            ["异常", "反常", "不正常", "值得关注", "信号", "背离"],
            ["37.6", "下降", "下滑", "减少"],
        ],
        expect_docs=["channel-ops", "q2-2026-review", "return-policy"],
        note="需要运营日历知识：9 月本是线上数码高峰，不升反降属异常信号",
    ),
    # ---------------------- 文档场景用例（第 2 场景回归） ----------------------
    # 共同设计原则：**必须由 writer 节点产出文档**，且文档内容只能来自素材。
    # 抓的是「文档能力被改坏」：writer 没被规划、素材没进提示词、结构缺失、数字编造。
    EvalCase(
        id="doc_q2_review_brief",
        scenario="document",
        question="请根据知识库中的材料，写一份面向管理层的 Q2 复盘简报。",
        expect_any=[
            # 素材里的核心事实（不引用素材写不出来）
            ["96", "达成率"],
            ["华东"],
        ],
        expect_docs=["q2-2026-review"],
        require_knowledge=True,
        expect_roles=["retriever", "writer", "reviewer"],
        note="文档场景：链路应为 检索→写作→审校，文档内容须来自复盘纪要",
    ),
    EvalCase(
        id="doc_return_policy_notice",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份面向客服团队的通知，"
            "说明线上渠道退货政策在 2026 年 8 月起的调整要点，"
            "以及客服接待用户时需要注意什么。"
        ),
        expect_any=[
            ["免运费", "运费"],
            ["质检", "全量", "审核"],
        ],
        expect_docs=["return-policy"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="文档场景第二例：体裁不同（通知、面向客服），同样是素材驱动写作",
    ),
    # ---------------------- 长期记忆用例（跨会话回归） ----------------------
    EvalCase(
        id="memory_recall_previous_conclusion",
        scenario="memory",
        question="上次分析华东区 9 月销售额下滑，当时的结论是什么？请回顾一下。",
        expect_any=[["20.8", "20.9", "下滑", "下降"]],
        # 结构性证据：记忆必须**真的被检索到**（清空记忆库该断言立即失败，
        # 与 RAG 用例的牙齿同一套方法——只查答案文本区分不出「真查了记忆」
        # 与「重新跑了一遍 SQL 恰好得到相同数字」）。
        expect_memories=["华东"],
        requires_case="core_decline_cause",
        # 前置用例必须**本次真的沉淀**（memory_sunk 非空）：开发库里积累的历史
        # 沉淀随时含「华东」，仅靠子串断言会在沉淀链路坏掉时假绿（独立复核确认）
        memory_source_case="core_decline_cause",
        # 只要求评审在场；不要求 analyst——口径/回顾类问题有时会被 Planner 规划到
        # 写作路径（writer→reviewer），那同样是合法链路，断言不应因此假失败
        expect_roles=["reviewer"],
        note="跨会话用例：依赖 core_decline_cause 先沉淀记忆（runner 自动补齐前置用例）",
    ),
    # ---------------------- 文档场景扩充（W8：每场景 ≥20） ----------------------
    # 设计原则：每条都必须 ①由 writer 产出文档、②素材驱动、③有可判定的素材锚点。
    # 素材只有 5 篇（约 1,500 字），靠**体裁 / 受众 / 跨文档组合**区分，避免同义改写。
    # 题面刻意避开「对外/发送/签发」等高风险关键词——内部材料不强制人工审批。
    EvalCase(
        id="doc_region_scope_training",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份面向新人的区域口径速查卡："
            "华东区包含哪些省市、区域归属按什么判定。"
        ),
        expect_any=[
            ["江苏", "浙江", "山东", "安徽"],
            ["收货地址", "注册地"],
        ],
        forbid=["八省", "六个省市"],
        expect_docs=["region-definition"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：新人速查卡；禁用词抓「八省/六省」这类编造",
    ),
    EvalCase(
        id="doc_cross_region_faq",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份跨区订单归属的常见问题解答（FAQ），"
            "说明跨区配送时订单算哪个区域、是否属于数据异常。"
        ),
        expect_any=[
            ["收货地址"],
            ["注册地"],
            ["大促", "跨区", "正常"],
        ],
        expect_docs=["region-definition"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：FAQ；锚点是「收货地址而非注册地」与「跨区属正常口径」",
    ),
    EvalCase(
        id="doc_region_comparability_note",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份跨区域对比的注意事项，"
            "说明销售额口径与区域口径是否一致、能否直接比较。"
        ),
        expect_any=[
            ["已支付", "支付", "不含"],
            ["一致", "可比", "可以直接比较", "同一口径"],
        ],
        expect_docs=["region-definition", "sales-data-dictionary"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="跨文档组合：区域口径 + 销售额口径",
    ),
    EvalCase(
        id="doc_return_policy_user_faq",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份关于线上退货政策调整的常见问题解答，"
            "说明 2026 年 8 月 15 日起的运费规则、退货时长与质检方式。"
        ),
        expect_any=[
            ["免运费"],
            ["7天", "3天"],
            ["质检", "全量"],
        ],
        forbid=["仍免运费", "继续免运费", "免运费仍然有效"],
        expect_docs=["return-policy"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：用户 FAQ；禁用词抓「免运费仍有效」这类反向编造",
    ),
    EvalCase(
        id="doc_return_policy_staff_training",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份面向客服团队的退货政策培训材料，"
            "列明接待要点与需要向用户解释的变化。"
        ),
        expect_any=[
            ["全量质检", "质检"],
            ["只退款", "不退货"],
            ["门店"],
        ],
        expect_docs=["return-policy"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：培训材料；锚点覆盖三项政策后果",
    ),
    EvalCase(
        id="doc_refund_rate_policy_memo",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份给管理层的备忘，"
            "解释 2026 年 9 月线上退货率上升是否意味着质量恶化，并给出判断依据。"
        ),
        expect_any=[
            ["4%", "5%"],
            ["口径", "政策", "非质量", "不是质量", "并非质量"],
        ],
        expect_docs=["return-policy"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：管理层备忘；核心判断与历史基线都来自政策文档",
    ),
    EvalCase(
        id="doc_refund_month_close_note",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份月末对账说明，"
            "解释退货流程变化如何影响退货金额的入账月份。"
        ),
        expect_any=[
            ["跨月", "次月"],
            ["3天", "7天"],
        ],
        expect_docs=["return-policy", "sales-data-dictionary"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="跨文档组合：流程时长 + refund_amount 当期口径",
    ),
    EvalCase(
        id="doc_store_vs_online_return_faq",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份内部 FAQ，"
            "说明门店渠道是否适用新的线上退货政策、两边的退货率是否可比。"
        ),
        expect_any=[
            ["门店"],
            ["不适用"],
            ["线上"],
        ],
        expect_docs=["return-policy"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：内部 FAQ；核心是「门店不适用」这一易错点",
    ),
    EvalCase(
        id="doc_q2_minutes",
        scenario="document",
        question=(
            "请根据知识库中的材料，整理一份 Q2 销售复盘的会议纪要要点："
            "整体达成情况、华东与华南的差异、以及当时的应对措施。"
        ),
        expect_any=[
            ["96%", "达成率"],
            ["华东"],
            ["华南"],
        ],
        expect_docs=["q2-2026-review"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：会议纪要（素材缺参会人/责任人，缺口应列入 material_gaps）",
    ),
    EvalCase(
        id="doc_east_online_digital_brief",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份关于华东区线上数码的专项简报，"
            "说明 Q2 的问题与 9 月大促期间的异常信号。"
        ),
        expect_any=[
            ["加购率", "12%"],
            ["竞品", "补贴"],
            ["9月", "大促"],
        ],
        expect_docs=["q2-2026-review", "channel-ops"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="跨文档组合：Q2 复盘 + 运营日历",
    ),
    EvalCase(
        id="doc_south_home_growth_brief",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份华南区增长经验的内部简报，"
            "说明增长来自哪个渠道与品类、以及可复制的原因。"
        ),
        expect_any=[
            ["华南"],
            ["家居"],
            ["复购"],
        ],
        expect_docs=["q2-2026-review"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="只讲素材中的增长来源，避免编造数字",
    ),
    EvalCase(
        id="doc_data_dictionary_training",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份 sales 表字段的培训材料，"
            "逐个说明核心字段的含义与口径。"
        ),
        expect_any=[
            ["stat_date", "快照"],
            ["revenue", "销售额"],
            ["refund_amount", "退货金额"],
        ],
        expect_docs=["sales-data-dictionary"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：字段培训；锚点是三个核心字段",
    ),
    EvalCase(
        id="doc_data_granularity_faq",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份数据粒度速查 FAQ，"
            "说明为什么基于 sales 表算不出日环比、日波动率这类指标。"
        ),
        expect_any=[
            ["月初", "每月1日", "月度"],
            ["不能", "无法", "不可"],
            ["日波动", "日环比", "日粒度"],
        ],
        forbid=["每日快照", "每天生成", "日快照"],
        expect_docs=["sales-data-dictionary", "region-definition"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：粒度 FAQ；禁用词抓「每日快照」这一反向编造",
    ),
    EvalCase(
        id="doc_refund_rate_training",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份退货率计算与口径的培训材料，"
            "说明退货率的计算方式与政策变化带来的口径影响。"
        ),
        expect_any=[
            ["refund_amount", "退货金额"],
            ["revenue", "销售额"],
            ["退货率"],
        ],
        expect_docs=["sales-data-dictionary", "return-policy"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="跨文档组合：公式来自字典、口径影响来自政策",
    ),
    EvalCase(
        id="doc_analyst_data_checklist",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份分析师取数前的自检清单，"
            "覆盖数据粒度、销售额口径与区域口径三类检查项。"
        ),
        expect_any=[
            ["快照", "每月1日"],
            ["口径", "不含"],
            ["日粒度", "日波动"],
        ],
        expect_docs=["sales-data-dictionary", "region-definition"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：自检清单；三类检查项各来自一篇素材",
    ),
    EvalCase(
        id="doc_promo_calendar_notice",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份渠道运营节奏的内部通知，"
            "说明线上与门店各月的大促安排以及错峰的目的。"
        ),
        expect_any=[
            ["3月", "6月", "9月", "11月"],
            ["10天", "提前"],
            ["门店"],
        ],
        expect_docs=["channel-ops"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：内部通知（非高风险，不触发审批）",
    ),
    EvalCase(
        id="doc_sep_promo_brief",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份 9 月大促背景简报，"
            "说明 9 月对线上数码意味着什么、以及不升反降为什么值得关注。"
        ),
        expect_any=[
            ["9月"],
            ["异常", "反常", "不升反降"],
            ["政策", "竞品", "大促"],
        ],
        expect_docs=["channel-ops", "q2-2026-review", "return-policy"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="三文档组合；不要求出现 37.6%（那是 SQL 数据不是语料）",
    ),
    EvalCase(
        id="doc_channel_comparison_guide",
        scenario="document",
        question=(
            "请根据知识库中的材料，写一份渠道对比的分析指南，"
            "说明线上与门店在客单价、订单量波动上的差异，以及大促期间要注意的渠道分流现象。"
        ),
        expect_any=[
            ["客单价", "订单量"],
            ["家居"],
            ["分流", "错开", "反向"],
        ],
        expect_docs=["channel-ops"],
        require_knowledge=True,
        expect_roles=["writer", "reviewer"],
        note="体裁：分析指南；锚点是渠道差异与分流现象",
    ),
    # ---------------------- 分析/RAG 扩充（W8：+5，总量 50） ----------------------
    EvalCase(
        id="honest_unknown_channel",
        question="华东区 9 月直播带货渠道的销售额是多少？",
        expect_any=[["没有", "无数据", "未找到", "不存在", "未覆盖", "0行", "不含", "无此"]],
        expect_insufficient=True,
        note="诚实性变体：数据里没有「直播」渠道，必须承认而不是编造",
    ),
    EvalCase(
        id="region_comparability_ratio",
        question="华东区和华南区的销售额可以直接比较吗？依据是什么？",
        expect_any=[
            ["口径"],
            ["一致", "可比", "可以直接比较"],
        ],
        expect_docs=["region-definition"],
        require_knowledge=True,
        note="口径类问题：可比性依据来自区域口径说明（区域按收货地址、口径沿用）",
    ),
    EvalCase(
        id="refund_rate_base_period",
        question="9 月线上退货率 10.04%，与今年之前的月度水平相比是否明显偏高？",
        expect_any=[
            ["4%", "5%"],
            ["偏高", "上升", "高于", "异常"],
        ],
        expect_docs=["return-policy"],
        require_knowledge=True,
        note="基期 4%–5% 只存在于政策文档；SQL 给不出「之前的正常水平」",
    ),
    EvalCase(
        id="data_granularity_boundary",
        question="能按周汇总 9 月的销售额吗？",
        expect_any=[
            ["不能", "无法", "不可", "做不到"],
            ["周", "月度", "快照", "月初"],
        ],
        expect_docs=["sales-data-dictionary"],
        require_knowledge=True,
        note="粒度边界：月度快照算不出周粒度（与日波动率为同类约束）",
    ),
    EvalCase(
        id="promo_month_compare",
        question="9 月和 6 月的线上数码销售额哪个更高？",
        expect_any=[["6月", "没有", "无数据", "缺失", "未覆盖"]],
        expect_insufficient=True,
        note="诚实性变体：数据只有 7–9 月，6 月不存在——必须声明而不是编造对比",
    ),
]


def get_cases(
    ids: list[str] | None = None,
    limit: int | None = None,
    scenario: str | None = None,
) -> list[EvalCase]:
    selected = CASES
    if ids:
        wanted = set(ids)
        selected = [c for c in CASES if c.id in wanted]
    if scenario:
        selected = [c for c in selected if c.scenario == scenario]
    if limit is not None:
        selected = selected[:limit]
    return list(selected)
