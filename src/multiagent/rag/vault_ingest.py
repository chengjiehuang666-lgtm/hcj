"""把 Obsidian 仓库（Markdown / docx）批量灌入项目知识库。

用法：
    python -m multiagent.rag.vault_ingest --vault "G:/第二大脑obsidian仓库/obsidian仓库" --dry-run
    python -m multiagent.rag.vault_ingest --vault "G:/第二大脑obsidian仓库/obsidian仓库"

分片：Markdown 按标题切段，每块保留标题路径；块内再按段落合并到约 800 字、120 字重叠。
安全：默认排除 `.obsidian`、附件目录、`多agent记忆/`（含项目路线图与评测答案，灌进来会污染业务检索）
和 `AGENTS.md`（Agent 指令，避免被当作知识注入）；正文过 W9 DLP 脱敏后再入库。
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

from ..config import Settings
from ..security.dlp import redact_text
from .chunking import Chunk, chunk_markdown, markdown_title, strip_frontmatter
from .embeddings import EmbeddingError
from .store import Document, KnowledgeStore, KnowledgeStoreError

TEXT_SUFFIXES = {".md", ".docx"}
EXCLUDED_FILES = {"AGENTS.md"}
EXCLUDED_DIRS = {".obsidian", "附件图片"}
PROJECT_MEMORY_DIR = "多agent记忆"


def _docx_text(path: Path) -> str:
    """不依赖 python-docx：从 docx 的 document.xml 提取段落文本。"""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", "ignore")
    xml = xml.replace("</w:p>", "\n").replace("<w:tab/>", "\t")
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def iter_vault_files(root: Path, *, include_project_memory: bool = False):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if any(part.startswith(".") for part in parts):
            continue
        if any(part in EXCLUDED_DIRS for part in parts):
            continue
        if not include_project_memory and PROJECT_MEMORY_DIR in parts:
            continue
        if rel.name in EXCLUDED_FILES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path, rel


def load_document(path: Path, rel: Path) -> tuple[Document, int]:
    """读取一个文件并做 DLP 脱敏；返回 (Document, 脱敏次数)。"""
    if path.suffix.lower() == ".docx":
        raw = _docx_text(path)
        title = path.stem
        body = raw
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        title = markdown_title(raw, path.stem)
        body = strip_frontmatter(raw)
    body, redactions = redact_text(body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    doc_id = "vault:" + rel.with_suffix("").as_posix()
    return Document(doc_id=doc_id, title=title.strip() or path.stem, text=body), redactions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m multiagent.rag.vault_ingest")
    parser.add_argument(
        "--vault",
        default=os.getenv("OBSIDIAN_VAULT_DIR"),
        help="Obsidian 仓库根目录（或设置 OBSIDIAN_VAULT_DIR）",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--limit", type=int, help="只处理前 N 个文件（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写库、不生成向量")
    parser.add_argument("--stats", action="store_true", help="只看当前知识库统计")
    parser.add_argument(
        "--include-project-memory",
        action="store_true",
        help="把 多agent记忆/ 也灌入（含路线图/评测答案，默认排除以免污染检索）",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    store = KnowledgeStore(settings)

    if args.stats:
        try:
            print("知识库统计:", store.stats())
            return 0
        except KnowledgeStoreError as exc:
            print(f"✗ 失败: {exc}", file=sys.stderr)
            return 1

    if not args.vault:
        print(
            "缺少 --vault。示例：\n"
            "  python -m multiagent.rag.vault_ingest --vault \"G:/第二大脑obsidian仓库/obsidian仓库\" --dry-run",
            file=sys.stderr,
        )
        return 2
    root = Path(args.vault)
    if not root.is_dir():
        print(f"✗ 仓库目录不存在: {root}", file=sys.stderr)
        return 2

    files = list(
        iter_vault_files(
            root, include_project_memory=args.include_project_memory
        )
    )
    if args.limit:
        files = files[: args.limit]

    started = time.perf_counter()
    documents: list[Document] = []
    chunks: list[Chunk] = []
    skipped: list[tuple[str, str]] = []
    redactions = 0
    per_dir: Counter[str] = Counter()

    for path, rel in files:
        try:
            doc, count = load_document(path, rel)
        except Exception as exc:  # noqa: BLE001 — 单文件失败不阻塞整批
            skipped.append((rel.as_posix(), f"{type(exc).__name__}: {exc}"[:160]))
            continue
        if not doc.text.strip():
            skipped.append((rel.as_posix(), "空文档"))
            continue
        if count:
            redactions += count
        documents.append(doc)
        per_dir[rel.parts[0] if len(rel.parts) > 1 else "(根目录)"] += 1
        chunks.extend(
            chunk_markdown(
                doc.doc_id,
                doc.title,
                doc.text,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        )

    print(f"▶ 扫描 {len(files)} 个文件，有效 {len(documents)} 篇，跳过 {len(skipped)} 篇")
    print(f"  分片：{len(chunks)} 块（chunk_size={args.chunk_size}, overlap={args.overlap}）")
    print(f"  字符：{sum(len(d.text) for d in documents):,}；DLP 脱敏 {redactions} 处")
    for name, count in per_dir.most_common():
        print(f"    {name}: {count} 篇")
    if skipped:
        print(f"  跳过示例：{skipped[:5]}")
    print(
        "  默认排除：.obsidian、附件图片、多agent记忆、AGENTS.md"
        + ("（已包含项目记忆）" if args.include_project_memory else "")
    )

    if args.dry_run:
        print("▶ dry-run：未写库")
        return 0
    if not chunks:
        print("✗ 没有可灌入的分块")
        return 1

    try:
        result = store.ingest_chunks(chunks, purge_prefix="vault:")
        print(
            f"✓ 完成：{result['documents']} 篇 / {result['chunks']} 块，"
            f"耗时 {time.perf_counter() - started:.1f}s"
        )
        print("  当前统计:", store.stats())
        return 0
    except (EmbeddingError, KnowledgeStoreError) as exc:
        print(f"✗ 失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
