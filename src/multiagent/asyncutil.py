"""Windows 上的 asyncio 事件循环兼容。

**背景**：psycopg 的异步模式明确不支持 Windows 默认的 `ProactorEventLoop`，
直接用 `asyncio.run()` 会报：

    Psycopg cannot use the 'ProactorEventLoop' to run in async mode

**约定**：凡是「异步 + PostgreSQL」的入口，都必须经由本模块的 `run_async()`，
不要在业务代码里直接调用 `asyncio.run()`。已由
`tests/test_async_loop_compat.py` 守护。

**FastAPI/uvicorn 用法**：uvicorn 自己管理事件循环，因此需要在**启动 uvicorn 之前**
调用 `install_event_loop_policy()`，例如放在应用入口模块的顶部。
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

IS_WINDOWS = sys.platform == "win32"


def install_event_loop_policy() -> None:
    """在启动任何异步运行时（如 uvicorn）之前调用，切换到兼容的事件循环。"""
    if IS_WINDOWS:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """执行协程；在 Windows 上强制使用 Selector 事件循环。"""
    if not IS_WINDOWS:
        return asyncio.run(coro)

    try:
        # Python 3.12+ 支持 loop_factory，无需改动全局策略，副作用最小
        return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
    except TypeError:
        # Python 3.11 没有 loop_factory 参数
        install_event_loop_policy()
        return asyncio.run(coro)
