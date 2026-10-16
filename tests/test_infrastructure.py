"""基础设施连通性冒烟测试：PostgreSQL(pgvector) 与 Redis。

若容器未启动则自动跳过，不阻塞契约与网关测试。
启动方式：docker compose up -d
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.config import Settings  # noqa: E402

settings = Settings(_env_file=ROOT / ".env")


def _pg_connect():
    import psycopg

    try:
        return psycopg.connect(settings.postgres_dsn, connect_timeout=5)
    except Exception as exc:  # 容器没起来就跳过
        pytest.skip(f"PostgreSQL 不可达（docker compose up -d 了吗？）: {exc}")


def _redis_client():
    import redis

    client = redis.from_url(settings.redis_url, socket_connect_timeout=5)
    try:
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis 不可达（docker compose up -d 了吗？）: {exc}")
    return client


# ----------------------------------------------------------------- PostgreSQL


def test_postgres_reachable_and_is_pg16() -> None:
    with _pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_setting('server_version_num')::int")
        version_num = cur.fetchone()[0]
    assert version_num >= 160000, f"期望 PostgreSQL 16+，实际 {version_num}"


def test_pgvector_extension_installed() -> None:
    with _pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    assert row is not None, "pgvector 扩展未安装（见 docker/initdb/01-extensions.sql）"
    assert row[0].startswith("0.8"), f"pgvector 版本异常: {row[0]}"


def test_pgvector_stores_and_searches_by_cosine_distance() -> None:
    """真正验证向量写入与相似度检索可用——长期记忆的核心能力。"""
    with _pg_connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE _v (id text, embedding vector(3)) ON COMMIT DROP")
        cur.execute(
            "INSERT INTO _v (id, embedding) VALUES (%s, %s), (%s, %s), (%s, %s)",
            ("a", "[1,0,0]", "b", "[0.9,0.1,0]", "c", "[0,1,0]"),
        )
        # 查询向量与 a 同向，期望 a 最近、c 最远
        cur.execute("SELECT id FROM _v ORDER BY embedding <=> %s::vector LIMIT 3", ("[1,0,0]",))
        order = [r[0] for r in cur.fetchall()]
    assert order[0] == "a" and order[-1] == "c", f"相似度排序异常: {order}"


def test_pg_trgm_available_for_hybrid_search() -> None:
    """混合检索里的关键词侧（BM25 的 PG 近似）依赖 pg_trgm。"""
    with _pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT similarity('销售额下滑', '销售额明显下滑')")
        score = cur.fetchone()[0]
    assert 0.0 < score <= 1.0, f"pg_trgm similarity 异常: {score}"


# ---------------------------------------------------------------------- Redis


def test_redis_reachable() -> None:
    client = _redis_client()
    assert client.ping() is True


def test_redis_read_write_roundtrip() -> None:
    client = _redis_client()
    key = "multiagent:smoke:test"
    try:
        client.set(key, "ok", ex=30)
        assert client.get(key) == b"ok"
    finally:
        client.delete(key)


def test_redis_stream_usable_for_blackboard_notifications() -> None:
    """黑板变更通知计划走 Redis Stream（见《系统架构设计》2.2.3）。"""
    client = _redis_client()
    stream = "multiagent:smoke:stream"
    try:
        msg_id = client.xadd(stream, {"event": "blackboard.updated", "key": "analyst_1.report"})
        assert msg_id, "xadd 未返回消息 ID"
        entries = client.xrange(stream)
        assert entries and entries[0][1][b"event"] == b"blackboard.updated"
    finally:
        client.delete(stream)
