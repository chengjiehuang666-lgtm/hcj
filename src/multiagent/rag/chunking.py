"""文档分块。

朴素但**可解释**的分块策略：优先按段落边界切，段落过长才硬切，
块之间保留一段重叠以免切断语义。刻意不用「按 token 数切」的复杂方案——
本项目的语料是内部业务文档，段落边界本身就是天然语义边界。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    title: str
    content: str


def _hard_split(text: str, size: int) -> list[str]:
    """超长段落按固定长度硬切（宁可切断也不要产生超长块）。"""
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    *,
    chunk_size: int = 320,
    overlap: int = 60,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正")
    overlap = max(0, min(overlap, chunk_size // 2))

    paragraphs = [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        pieces.extend(_hard_split(para, chunk_size) if len(para) > chunk_size else [para])

    chunks: list[str] = []
    current = ""
    pending = list(pieces)
    while pending:
        piece = pending.pop(0)
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # 放不下：先把当前块收尾，再开新块
        tail = ""
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = ""

        # 关键：overlap 尾巴会占用本块预算，正文只能用剩下的部分。
        # 早先漏了这一步，导致 tail + piece 超过 chunk_size。
        budget = chunk_size - (len(tail) + 1 if tail else 0)
        if budget <= 0:  # overlap 配置过大时退化为不带 overlap 的新块
            tail, budget = "", chunk_size
        if len(piece) > budget:
            head, rest = piece[:budget], piece[budget:]
            current = f"{tail}\n{head}" if tail else head
            pending.insert(0, rest)
        else:
            current = f"{tail}\n{piece}" if tail else piece

    if current.strip():
        chunks.append(current)

    return [
        Chunk(doc_id=doc_id, chunk_index=i, title=title, content=content)
        for i, content in enumerate(chunks)
    ]
