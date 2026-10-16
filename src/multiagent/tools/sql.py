"""只读 SQL 执行工具 —— Agent 访问数据库的唯一入口（SQL 沙箱）。

安全设计为**三层防御**，任何一层单独失效都不会导致写库：

1. **数据库角色层（最强）**：使用 `analyst_ro` 只读角色，仅有 SELECT 权限，
   且该角色被设置了 `statement_timeout = 5s`。即使下面两层被绕过，写操作在
   数据库侧也会被拒绝。已实测 INSERT/UPDATE/DELETE/DROP/CREATE 全部返回
   permission denied。
2. **语句校验层（本模块）**：单语句、必须以 SELECT/WITH 开头、剔除字符串字面量
   后扫描禁用关键字。
3. **结果规模层（本模块）**：把查询包成子查询并强制 LIMIT，避免一次拉回海量数据。

调用方拿到结果时同时得到 `sql_hash`，用于《产品需求》要求的结论溯源。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from ..config import Settings


class SqlToolError(Exception):
    """SQL 工具错误基类。"""


class UnsafeSqlError(SqlToolError):
    """语句未通过安全校验。"""


class SqlExecutionError(SqlToolError):
    """语句通过校验但执行失败（语法错误、超时、权限不足等）。"""


# 剔除字符串字面量与注释后再扫描，避免 WHERE note = 'delete' 之类的误报。
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum"
    r"|reindex|call|do|set|reset|comment|lock|refresh|merge|upsert|commit|rollback"
    r"|begin|savepoint|listen|notify|discard|analyze|cluster|security|prepare"
    r"|execute|declare|fetch|move|import|pg_read_file|pg_ls_dir|lo_import|dblink)\b",
    re.IGNORECASE,
)
_ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def validate_sql(sql: str) -> str:
    """校验并返回规范化后的语句；不合法则抛 UnsafeSqlError。"""
    if not sql or not sql.strip():
        raise UnsafeSqlError("SQL 为空")

    cleaned = _BLOCK_COMMENT.sub(" ", sql)
    cleaned = _LINE_COMMENT.sub(" ", cleaned)

    # 去除末尾分号后，任何剩余分号都意味着多语句
    body = cleaned.strip().rstrip(";").strip()
    if ";" in body:
        raise UnsafeSqlError("只允许执行单条语句（检测到多个分号分隔的语句）")

    if not _ALLOWED_START.match(body):
        raise UnsafeSqlError("只允许 SELECT 或 WITH 开头的只读查询")

    scannable = _STRING_LITERAL.sub("''", body)
    if hit := _FORBIDDEN.search(scannable):
        raise UnsafeSqlError(f"语句包含被禁止的关键字: {hit.group(0).upper()}")

    return body


def _jsonify(value: Any) -> Any:
    """把 psycopg 返回的不可 JSON 序列化类型转成基础类型（Decimal/date 等）。"""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    return value


@dataclass
class SqlResult:
    """一次只读查询的结果。"""

    sql: str
    sql_hash: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    limited: bool = False
    duration_ms: int = 0

    def as_prompt_text(self, max_rows: int = 50) -> str:
        """转成给模型看的紧凑文本，避免把整张结果集塞进 prompt。"""
        if not self.rows:
            return "（查询无返回行）"
        head = self.rows[:max_rows]
        lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in head]
        if len(self.rows) > max_rows:
            lines.append(f"...（另有 {len(self.rows) - max_rows} 行已省略）")
        return "\n".join(lines)


class SqlSandbox:
    """通过只读角色执行 SELECT 的沙箱。"""

    def __init__(self, settings: Settings) -> None:
        self._dsn = settings.postgres_ro_dsn
        self._max_rows = settings.sql_max_rows
        self._timeout = settings.sql_timeout_seconds

    # ---- 内部 ----
    def _wrap(self, sql: str) -> str:
        """包一层子查询强制 LIMIT，保证返回行数有上限。"""
        return (
            "SELECT * FROM (\n"
            f"{sql}\n"
            f") AS _sandbox_subquery LIMIT {self._max_rows}"
        )

    # ---- 对外 ----
    def execute(self, sql: str) -> SqlResult:
        import psycopg
        from psycopg.rows import dict_row

        body = validate_sql(sql)
        wrapped = self._wrap(body)
        sql_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

        started = time.perf_counter()
        try:
            with psycopg.connect(
                self._dsn, connect_timeout=self._timeout, row_factory=dict_row
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(wrapped)
                    raw = cur.fetchall()
        except Exception as exc:
            raise SqlExecutionError(f"查询执行失败: {exc}") from exc
        duration_ms = int((time.perf_counter() - started) * 1000)

        rows = [{k: _jsonify(v) for k, v in row.items()} for row in raw]
        return SqlResult(
            sql=body,
            sql_hash=sql_hash,
            rows=rows,
            row_count=len(rows),
            limited=len(rows) >= self._max_rows,
            duration_ms=duration_ms,
        )

    def describe_schema(self) -> list[dict[str, Any]]:
        """返回可查询的表与列，供检索 Agent 判断「有哪些数据可用」。"""
        import psycopg
        from psycopg.rows import dict_row

        query = """
            SELECT c.table_name, c.column_name, c.data_type
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON t.table_schema = c.table_schema AND t.table_name = c.table_name
            WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'
            ORDER BY c.table_name, c.ordinal_position
        """
        try:
            with psycopg.connect(
                self._dsn, connect_timeout=self._timeout, row_factory=dict_row
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    return [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            raise SqlExecutionError(f"读取表结构失败: {exc}") from exc

    def schema_as_prompt_text(self) -> str:
        grouped: dict[str, list[str]] = {}
        for col in self.describe_schema():
            grouped.setdefault(col["table_name"], []).append(
                f"{col['column_name']} {col['data_type']}"
            )
        return "\n".join(
            f"- {table}({', '.join(cols)})" for table, cols in grouped.items()
        )
