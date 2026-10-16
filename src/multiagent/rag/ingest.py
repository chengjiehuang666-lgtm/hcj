"""知识库灌库入口。

用法：
    python -m multiagent.rag.ingest            # 灌入内置演示语料
    python -m multiagent.rag.ingest --stats    # 只看当前知识库统计

注意：表结构由 `docker/initdb/03-knowledge-schema.sql` 建立，
若数据卷被重建（docker compose down -v）需要重新执行本命令。
"""

from __future__ import annotations

import argparse
import sys

from ..config import Settings
from ..rag.corpus import DOCUMENTS
from ..rag.embeddings import EmbeddingError
from ..rag.store import KnowledgeStore, KnowledgeStoreError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m multiagent.rag.ingest")
    parser.add_argument("--stats", action="store_true", help="只输出知识库统计，不灌库")
    args = parser.parse_args(argv)

    settings = Settings()
    store = KnowledgeStore(settings)

    try:
        if args.stats:
            print("知识库统计:", store.stats())
            return 0

        print(f"▶ 灌库：{len(DOCUMENTS)} 篇文档")
        print(f"  embedding 模型: {settings.embedding_model}"
              f"（缓存 {settings.embedding_cache_dir}，镜像 {settings.hf_endpoint}）")
        result = store.ingest(DOCUMENTS)
        print(f"✓ 完成：{result['documents']} 篇 / {result['chunks']} 个分块")
        print("  当前统计:", store.stats())
        return 0
    except (EmbeddingError, KnowledgeStoreError) as exc:
        print(f"✗ 失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
