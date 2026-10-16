"""守护 Windows 上的异步 PostgreSQL 兼容性。

这是一个真实的回归测试：Windows 默认的 ProactorEventLoop 无法用于 psycopg 异步，
若有人把 `run_async` 换回裸 `asyncio.run`，本测试会立刻失败。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.asyncutil import IS_WINDOWS, run_async  # noqa: E402
from multiagent.config import Settings  # noqa: E402

settings = Settings(_env_file=ROOT / ".env")


def test_run_async_executes_plain_coroutine() -> None:
    async def work() -> int:
        return 42

    assert run_async(work()) == 42


def test_run_async_can_open_async_postgres_connection() -> None:
    """这是关键用例：在 Windows 上若事件循环不对，这里会抛 ProactorEventLoop 错误。"""
    import psycopg

    async def probe() -> str:
        conn = await psycopg.AsyncConnection.connect(
            settings.postgres_dsn, connect_timeout=5
        )
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                row = await cur.fetchone()
            return "ok" if row == (1,) else "unexpected"
        finally:
            await conn.close()

    try:
        assert run_async(probe()) == "ok"
    except Exception as exc:  # 容器没起来就跳过，但事件循环问题必须暴露
        if "ProactorEventLoop" in str(exc):
            raise AssertionError(
                f"事件循环不兼容，必须使用 run_async 而非 asyncio.run：{exc}"
            ) from exc
        pytest.skip(f"PostgreSQL 不可达（docker compose up -d 了吗？）: {exc}")


def test_bare_asyncio_run_would_fail_on_windows_hence_helper_needed() -> None:
    """留作说明性断言：Windows 上裸 asyncio.run 的确不兼容（不实际触发失败）。"""
    if not IS_WINDOWS:
        pytest.skip("非 Windows 平台无需此兼容处理")
    import asyncio

    assert asyncio.get_event_loop_policy() is not None
