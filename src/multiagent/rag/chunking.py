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


# ---------------------------------------------------------------- Markdown 分片

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def strip_frontmatter(text: str) -> str:
    """去掉 YAML frontmatter（只用于灌库正文，不修改源文件）。"""
    return _FRONTMATTER.sub("", text, count=1)


def markdown_title(text: str, fallback: str) -> str:
    """优先 frontmatter.title，其次第一个一级标题，最后文件名。"""
    front = _FRONTMATTER.match(text)
    if front:
        match = re.search(r"^title:\s*(.+)$", front.group(0), re.MULTILINE)
        if match:
            return match.group(1).strip().strip("\"'")
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def _markdown_sections(body: str) -> list[tuple[str, str]]:
    """按 ATX 标题切段，返回 (标题路径, 正文)。代码围栏里的 # 不算标题。"""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    trail = ""
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal current
        text = "\n".join(current).strip()
        if text:
            sections.append((trail, text))
        current = []

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            trail = " > ".join(item for _, item in stack)
        else:
            current.append(line)
    flush()
    return sections


def chunk_markdown(
    doc_id: str,
    title: str,
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Markdown 标题感知分片：每个块带标题路径，块内再按段落合并。

    比「整篇按段落切」更适合 Obsidian 笔记：长文里的章节边界就是语义边界，
    标题路径同时进入 `title` 与正文前缀，检索时更容易命中上下文。
    """
    body = strip_frontmatter(text)
    sections = _markdown_sections(body)
    if not sections:
        sections = [("", body)]

    chunks: list[Chunk] = []
    for trail, section in sections:
        section_title = f"{title} · {trail}" if trail else title
        section_title = section_title[:200]
        content = f"{trail}\n{section}" if trail else section
        for part in chunk_document(
            doc_id,
            section_title,
            content,
            chunk_size=chunk_size,
            overlap=overlap,
        ):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=len(chunks),
                    title=part.title,
                    content=part.content,
                )
            )
    return chunks
