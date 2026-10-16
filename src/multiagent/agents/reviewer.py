"""评审 Agent（Reviewer）：对上游 Agent 的产物做交叉校验。

职责（产品需求 §1.3）：识别幻觉、逻辑错误、安全与合规问题——**逐条核对**上游
节点声明的验收标准，并检查每条结论是否有可回溯的证据。

关键设计（每条都有明确理由）：

- **只给证据、不给期望答案**：评审的输入是上游真实产物与来源明细。给它「正确答案」
  会让它变成迎合者而不是核对者（与归因环节不接触验收标准是同一条纪律）。
- **verdict 三值且判定规则可测**：pass / pass_with_issues / fail。fail 的语义被限缩为
  「存在 unmet 的验收标准或 blocker 级问题」——它是**长期记忆沉淀门槛**的依据
  （fail 的结论不进记忆库），语义放大会让整条沉淀链路静默失效。
- **状态流转走黑板乐观锁**：审查结论以 `blackboard_updates`（提交读到的 version）
  把上游条目从 draft 流转为 validated / rejected——这是契约里条目状态与版本锁的
  第一个真实使用场景。
- **不产出 `summary` 键**：最终结论仍是 analyst 的产物（汇点取数逻辑会退化到它的
  summary），评审意见是独立字段，不得顶替业务结论。
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType
from ..gateway import ROUTE_REVIEWER
from .base import AgentContext, BaseAgent, _Usage

_SYSTEM_PROMPT = """你是评审 Agent（Reviewer）。你的任务是对上游 Agent 的产物做**交叉校验**：
逐条核对验收标准，检查每条结论是否有可回溯的证据，识别幻觉与逻辑错误。

硬性要求：
1. 对每条验收标准给出结果：met（满足）/ unmet（未满足）/ unverifiable（无法判定），
   并在 evidence 里给出**具体证据**（引用某行数字、行数、SQL 哈希或来源标注）。
   - 若验收标准要求的材料**没有出现在给定产物里**，记 unverifiable 并在 evidence
     说明缺了什么——不得记为 unmet（材料缺失不是上游的失败，是评审自身的输入不足）；
   - 只有材料齐备、但标准确实没有被满足时，才记 unmet。
2. 核对方式（按产物里实际出现的字段）：
   - 结论中的每个数字必须能在查询结果（rows_preview / 行数）中找到出处，
     或有明确的来源标注（如引用背景资料时注明文档标题）；
   - 若结论声明数据不足（insufficient_data），检查该声明是否合理——是诚实说明
     确实缺数据，还是回避了本可回答的问题；
   - 若产物引用了背景资料，检查它没有把背景资料里的数字当作查询结果引用，
     且引用的事实确实出现在上游产物的知识片段里；
   - 上游有多个节点时，核对它们之间是否一致（如同一指标出现两个值）。
3. issues 按严重度分级，描述控制在一句话内：
   - blocker：结论无证据支撑、数字疑似编造、明显答非所问、违反了应遵守的约束；
   - major：部分验收标准未满足，或证据薄弱；
   - minor：表述、完整性等小问题。
4. verdict 判定规则（严格遵守，不得放宽）：
   - 存在 unmet 的验收标准，或任何 blocker 级问题 → "fail"
   - 验收标准没有 unmet、但有 major/minor 问题（或存在 unverifiable）→ "pass_with_issues"
   - 全部 met 且无问题 → "pass"
5. 你**不得**引入任何新的数字或事实；evidence 只能引用给定材料。
6. confidence 是 0-1 的数字，表示你对评审结论的把握程度。
7. 输出单个 JSON 对象，不要 markdown 围栏，不要解释文字：
   {"verdict": "pass|pass_with_issues|fail",
    "checks": [{"criterion": "...", "result": "met|unmet|unverifiable", "evidence": "..."}],
    "issues": [{"severity": "blocker|major|minor", "description": "..."}],
    "confidence": 0.8}"""

#: 从合并输入里排除的保留键——兜底视图（上游没写黑板条目时）不该把它们当产物。
_RESERVED_INPUT_KEYS = frozenset(
    {"question", "now", "acceptance", "node_id", "upstream_nodes", "blackboard",
     "upstream_sources"}
)


class _Check(BaseModel):
    criterion: str
    result: Literal["met", "unmet", "unverifiable"]
    evidence: str = ""


class _Issue(BaseModel):
    severity: Literal["blocker", "major", "minor"]
    description: str


class _Verdict(BaseModel):
    verdict: Literal["pass", "pass_with_issues", "fail"]
    checks: list[_Check] = Field(default_factory=list)
    issues: list[_Issue] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)


def effective_verdict(review: _Verdict) -> str:
    """按**代码**落实 verdict 判定规则，而不是只写在提示词里。

    独立复核发现：模型可能输出自相矛盾的评审（verdict=pass 却带着 unmet 或
    blocker），旧的实现照单全收并据此把条目标为 validated——而 verdict 又决定
    长期记忆的沉淀门槛，错版会一路传下去。规则以代码为准：

    - 任一 unmet 或任一 blocker → fail
    - 否则有 major/minor 问题、或存在 unverifiable → pass_with_issues
    - 否则 pass
    """
    if any(c.result == "unmet" for c in review.checks) or any(
        i.severity == "blocker" for i in review.issues
    ):
        return "fail"
    if review.issues or any(c.result == "unverifiable" for c in review.checks):
        return "pass_with_issues"
    return "pass"


def _field_view(key: str, value: Any) -> Any:
    """按字段裁剪大对象：行预览只留前 20 行，知识片段保留出处、正文截断。"""
    if key == "rows_preview" and isinstance(value, list):
        return value[:20]
    if key == "knowledge" and isinstance(value, list):
        return [
            {
                **{k: v for k, v in item.items() if k != "content"},
                "content": str(item.get("content", ""))[:400],
            }
            for item in value
            if isinstance(item, dict)
        ]
    return value


def _compact(outputs: dict[str, Any], *, per_field: int = 2500) -> str:
    """把节点产物压成提示词可读的 JSON；单字段超长则截断并注明。"""
    view: dict[str, Any] = {}
    for key, value in outputs.items():
        value = _field_view(str(key), value)
        text = json.dumps(value, ensure_ascii=False, default=str)
        view[key] = value if len(text) <= per_field else text[:per_field] + "…（已截断）"
    return json.dumps(view, ensure_ascii=False, default=str)


class ReviewerAgent(BaseAgent):
    role = "reviewer"
    version = "0.1.0"

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()

        question = inv.input.get("question", "")
        acceptance = [str(a) for a in (inv.input.get("acceptance") or [])]
        upstream_nodes = [str(n) for n in (inv.input.get("upstream_nodes") or [])]
        blackboard: dict[str, dict] = inv.input.get("blackboard") or {}
        upstream_sources: dict[str, list] = inv.input.get("upstream_sources") or {}

        if not upstream_nodes:
            return self.failed(
                inv,
                code="BAD_INPUT",
                message="reviewer 没有任何上游节点可核对（depends_on 为空）",
                retryable=False,
                usage=usage,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        # 产物按**黑板条目**逐节点呈现：比合并视图多带出处与版本，且能区分同名键
        # （两个节点都产出 key_findings 时，合并视图会丢失前者的归属）。
        sections: list[str] = []
        for key, entry in sorted(blackboard.items()):
            written_by = entry.get("written_by") or {}
            node_id = str(written_by.get("node_id") or key.split(".", 1)[0])
            sections.append(
                f"### 节点 {node_id} 的产物（黑板条目 {key} v{entry.get('version')}，"
                f"状态 {entry.get('status')}）\n{_compact(entry.get('payload') or {})}"
            )
        if not sections:
            # 兜底：上游没写黑板条目时用合并视图（正常链路不会走到这里）
            merged_view = {
                k: v for k, v in inv.input.items() if k not in _RESERVED_INPUT_KEYS
            }
            sections.append(f"### 上游产物（合并视图）\n{_compact(merged_view)}")

        evidence_lines = []
        for node_id in sorted(upstream_sources):
            refs = "；".join(
                f"{s.get('type')}:{s.get('ref')}"
                for s in (upstream_sources.get(node_id) or [])
                if isinstance(s, dict)
            )
            evidence_lines.append(f"- {node_id}: {refs or '（无来源记录）'}")

        acceptance_text = "\n".join(f"- {a}" for a in acceptance) or "（未提供）"
        user = (
            f"业务问题：{question}\n\n"
            f"需要你逐条核对的验收标准：\n{acceptance_text}\n\n"
            f"上游节点产物：\n\n" + "\n\n".join(sections) + "\n\n"
            f"上游证据明细（来源引用）：\n" + ("\n".join(evidence_lines) or "（无）")
        )

        try:
            review = await self.call_json(
                ctx,
                system=_SYSTEM_PROMPT,
                user=user,
                model_cls=_Verdict,
                usage=usage,
                route=ROUTE_REVIEWER,
            )
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

        # 审查结论经乐观锁流转上游条目状态：fail → rejected，否则 validated。
        # verdict 以**代码规则**为准（effective_verdict），模型的自报值单独留档：
        # 两者不一致说明模型输出了自相矛盾的评审，值得被发现。
        verdict = effective_verdict(review)
        target_status = "rejected" if verdict == "fail" else "validated"
        updates = [
            {
                "key": key,
                "expected_version": int(entry.get("version") or 1),
                "status": target_status,
            }
            for key, entry in sorted(blackboard.items())
        ]

        sources = [
            SourceRef(type=SourceType.model_inference, ref=self.labeled_version),
            *[
                SourceRef(
                    type=SourceType.document,
                    ref=f"黑板:{key}@v{entry.get('version')}",
                )
                for key, entry in sorted(blackboard.items())
            ],
        ]

        return self.ok(
            inv,
            outputs={
                "verdict": verdict,
                # 模型自报的 verdict 单独留档：与规则不一致时便于发现模型输出问题
                "model_verdict": review.verdict,
                "checks": [c.model_dump() for c in review.checks],
                "issues": [i.model_dump() for i in review.issues],
                "issue_count": len(review.issues),
                "reviewed_nodes": sorted(set(upstream_nodes)),
                "blackboard_updates": updates,
            },
            sources=sources,
            usage=usage,
            duration_ms=int((time.perf_counter() - started) * 1000),
            confidence=review.confidence,
            blackboard_writes=[f"{inv.node_id}.review"],
        )
