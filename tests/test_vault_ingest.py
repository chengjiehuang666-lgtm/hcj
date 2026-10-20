"""Obsidian 批量灌库：Markdown 标题分片、排除规则、DLP、docx 提取。"""

from __future__ import annotations

import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiagent.rag.chunking import chunk_markdown, markdown_title, strip_frontmatter  # noqa: E402
from multiagent.rag.vault_ingest import (  # noqa: E402
    _docx_text,
    iter_vault_files,
    load_document,
)

MD = """---
title: 测试笔记
tags: [a, b]
---

开场白。

# 一级标题

## 第一节

第一节的正文。手机 13812345678。

```python
# 这是代码注释，不是标题
print("x")
```

## 第二节

第二节的正文。
"""


def test_frontmatter_and_title() -> None:
    assert markdown_title(MD, "fallback") == "测试笔记"
    body = strip_frontmatter(MD)
    assert "title:" not in body
    assert "开场白" in body


def test_markdown_chunks_are_heading_aware() -> None:
    chunks = chunk_markdown("vault:x", "测试笔记", MD, chunk_size=200, overlap=20)
    assert chunks
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    titles = " ".join(c.title for c in chunks)
    assert "第一节" in titles and "第二节" in titles
    assert any("开场白" in c.content for c in chunks)
    # 代码块里的 # 注释不能被当成标题切段
    assert any("# 这是代码注释" in c.content for c in chunks)
    assert all(len(c.content) <= 200 for c in chunks)


def test_iter_vault_files_excludes_system_and_project_memory(tmp_path: pathlib.Path) -> None:
    (tmp_path / "The wiki").mkdir()
    (tmp_path / "The wiki" / "a.md").write_text("# A\n内容", encoding="utf-8")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "x.md").write_text("secret", encoding="utf-8")
    (tmp_path / "多agent记忆").mkdir()
    (tmp_path / "多agent记忆" / "roadmap.md").write_text("答案", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("规则", encoding="utf-8")

    default = [rel.as_posix() for _, rel in iter_vault_files(tmp_path)]
    assert default == ["The wiki/a.md"]
    with_memory = [
        rel.as_posix()
        for _, rel in iter_vault_files(tmp_path, include_project_memory=True)
    ]
    assert "多agent记忆/roadmap.md" in with_memory
    assert "AGENTS.md" not in with_memory


def test_load_document_redacts_pii(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "note.md"
    path.write_text(MD, encoding="utf-8")
    doc, redactions = load_document(path, pathlib.Path("note.md"))
    assert doc.doc_id == "vault:note"
    assert doc.title == "测试笔记"
    assert "13812345678" not in doc.text
    assert "[手机号已脱敏]" in doc.text
    assert redactions >= 1


def test_docx_text_extraction(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "q.docx"
    xml = (
        '<?xml version="1.0"?><w:document xmlns:w="w">'
        "<w:body><w:p><w:r><w:t>第一段</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>第二段</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    text = _docx_text(path)
    assert "第一段" in text and "第二段" in text
