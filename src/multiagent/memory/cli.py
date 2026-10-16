"""长期记忆维护命令行：统计、检索、清空。

用法：
    python -m multiagent.memory --stats                 # 默认：统计
    python -m multiagent.memory --search "华东区下滑"    # 混合检索预览
    python -m multiagent.memory --clear --yes           # 清空（破坏性，须显式 --yes）
"""

from __future__ import annotations

import argparse
import sys

from ..config import Settings
from .store import MemoryStore, MemoryStoreError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m multiagent.memory")
    parser.add_argument("--stats", action="store_true", help="打印记忆库统计（默认）")
    parser.add_argument("--search", metavar="QUERY", help="混合检索并打印命中")
    parser.add_argument("--clear", action="store_true", help="清空记忆库（破坏性）")
    parser.add_argument("--yes", action="store_true", help="确认破坏性操作")
    parser.add_argument("--top-k", type=int, default=None, help="检索条数上限")
    args = parser.parse_args(argv)

    settings = Settings()
    store = MemoryStore(settings)
    try:
        if args.clear:
            if not args.yes:
                print("拒绝执行：--clear 是破坏性操作，请附加 --yes 确认")
                return 2
            deleted = store.clear()
            print(f"已清空 {deleted} 条记忆")
            return 0

        if args.search:
            hits = store.search(args.search, top_k=args.top_k)
            if not hits:
                print("（无命中）")
                return 0
            for hit in hits:
                print(f"· [v{hit.memory_id}] {hit.score:.3f} | {hit.source_ref}")
                print(f"  {hit.content[:200]}")
            return 0

        # 默认打印统计
        stats = store.stats()
        print(f"长期记忆：共 {stats['total']} 条（validated {stats['validated']}）")
        return 0
    except MemoryStoreError as exc:
        print(f"操作失败：{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
