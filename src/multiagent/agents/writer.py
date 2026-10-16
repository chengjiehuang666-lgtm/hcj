"""写作 Agent（Writer）：按素材撰写正式文档。

第 2 个场景「报告/文档自动化」的核心角色（产品需求 §1.3：写作 Agent 属阶段二）。
链路形态：`retriever(knowledge) → writer → reviewer`。

关键纪律（与 Analyst 同源——「把模型从编数字的路径上隔离出去」）：

- **数字与事实只能来自素材**，并在文中标注出处；素材没有的写「素材未涵盖」，
  列进 `material_gaps`——缺口要显式暴露，而不是用流畅的文字掩盖；
- **不产出 `summary` 键**：文档正文才是产物本体，汇点取数不得被顶替。
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from ..contracts import AgentInvocation, AgentResult, SourceRef, SourceType
from ..gateway import ROUTE_WRITER
from .base import AgentContext, BaseAgent, _Usage

_SYSTEM_PROMPT = """你是写作 Agent（Writer）。根据给定素材撰写一份正式文档（Markdown）。

硬性要求：
1. **数字与事实只能来自下方素材**。引用素材内容时在句末标注来源，例如
   「（来源：《线上渠道退货政策调整通知》）」。
2. 素材未涵盖的信息：正文里**不得编造**；把缺口列进 material_gaps，
   正文相关位置可以写「（素材未涵盖）」。
3. 结构完整：一级标题（# 标题）、一段摘要、正文按主题分 2-4 节（## 小节），
   最后给出结论或建议。
4. 语言面向需求里说明的受众（如「面向管理层」要简明、先说结论；
   「面向客服团队」要可操作、分条列出要点）。
5. 若素材明显不足以成文，仍尽力产出一份最小文档，并把 material_gaps 列全。
6. 输出单个 JSON 对象，不要 markdown 围栏，不要解释文字；
   document_markdown 里用 \\n 表示换行：
   {"title": "...", "document_markdown": "# ...\\n\\n...",
    "used_sources": ["《…》"], "material_gaps": ["…"]}"""


class _Document(BaseModel):
    title: str = Field(min_length=1)
    document_markdown: str = Field(min_length=1)
    used_sources: list[str] = Field(default_factory=list)
    material_gaps: list[str] = Field(default_factory=list)


def _materials_text(
    knowledge: list[dict], memories: list[dict], *, limit: int = 6, per_chunk: int = 600
) -> str:
    """素材区：知识片段（写作的主要依据）+ 历史结论（仅供承接）。

    比分析场景的预算宽：写作就是要把素材用足，截太狠会让文档失血。
    """
    lines: list[str] = []
    for item in knowledge[:limit]:
        content = str(item.get("content", "")).replace("\n", " ")[:per_chunk]
        lines.append(f"《{item.get('title', '?')}》：{content}")
    for item in memories[:3]:
        content = str(item.get("content", "")).replace("\n", " ")[:500]
        lines.append(f"（历史结论，来源：{item.get('source_ref', '?')}）{content}")
    return "\n\n".join(lines) if lines else "（未检索到任何素材）"


class WriterAgent(BaseAgent):
    role = "writer"
    version = "0.1.0"

    async def invoke(self, inv: AgentInvocation, ctx: AgentContext) -> AgentResult:
        started = time.perf_counter()
        usage = _Usage()

        question = inv.input.get("question", "")
        acceptance = [str(a) for a in (inv.input.get("acceptance") or [])]
        knowledge = inv.input.get("knowledge", []) or []
        memories = inv.input.get("memories", []) or []

        acceptance_text = "\n".join(f"- {a}" for a in acceptance) or "（未提供）"
        user = (
            f"写作需求：{question}\n\n"
            f"本次需要满足的验收标准：\n{acceptance_text}\n\n"
            f"可用素材：\n{_materials_text(knowledge, memories)}"
        )

        try:
            doc = await self.call_json(
                ctx,
                system=_SYSTEM_PROMPT,
                user=user,
                model_cls=_Document,
                usage=usage,
                route=ROUTE_WRITER,
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

        sources = [SourceRef(type=SourceType.model_inference, ref=self.labeled_version)]
        sources.extend(
            SourceRef(
                type=SourceType.document,
                ref=f"知识库:{item.get('doc_id')}#{item.get('chunk_index')}",
            )
            for item in knowledge[:6]
        )
        if not knowledge:
            # 契约要求 status=ok 必带来源；「没有素材」本身也要如实记录
            sources.append(
                SourceRef(type=SourceType.tool, ref="素材检索无命中（文档缺口已列入 material_gaps）")
            )

        return self.ok(
            inv,
            outputs={
                "title": doc.title,
                "document_markdown": doc.document_markdown,
                "section_count": sum(
                    1
                    for line in doc.document_markdown.splitlines()
                    if line.startswith("## ")
                ),
                "char_count": len(doc.document_markdown),
                "used_sources": doc.used_sources,
                "material_gaps": doc.material_gaps,
            },
            sources=sources,
            usage=usage,
            duration_ms=int((time.perf_counter() - started) * 1000),
            # 没有素材时置信度显著降低：不可靠的文档不该伪装成可靠的
            confidence=0.8 if knowledge else 0.3,
            blackboard_writes=[f"{inv.node_id}.document"],
        )
