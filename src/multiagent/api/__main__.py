"""启动 HTTP 服务：`python -m multiagent.api`

**为什么不用 `install_event_loop_policy()`**：uvicorn 0.54 在 Windows 上**硬编码**
选择 ProactorEventLoop（见 `.venv/Lib/site-packages/uvicorn/loops/asyncio.py`：
`if sys.platform == "win32" and not use_subprocess: return asyncio.ProactorEventLoop`），
它会把自己的 loop factory 交给 `asyncio.Runner`，因此**提前设置全局策略无效**。

而 psycopg 的异步模式不支持 ProactorEventLoop（LangGraph 的异步 checkpointer 会用到），
所以必须显式覆盖 loop 工厂。`uvicorn.Config.get_loop_factory()` 对不在预设表里的值
走 `import_from_string` 并直接返回，因此传 `"asyncio:SelectorEventLoop"` 即可。
"""

from __future__ import annotations

import argparse
import sys

from .app import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m multiagent.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    kwargs: dict = {}
    if sys.platform == "win32":
        # 覆盖 uvicorn 的 Windows 默认值，否则流水线里的 psycopg 异步连接会失败
        kwargs["loop"] = "asyncio:SelectorEventLoop"

    uvicorn.run(create_app(), host=args.host, port=args.port, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
