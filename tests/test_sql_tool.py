"""SQL 沙箱测试：安全约束必须逐条成立。

沙箱是「Agent 不能写库」这条底线的实现，因此这些测试重点在**拒绝**而非接受。
只读角色在数据库侧的第二重验证由 tests/test_infrastructure.py 覆盖。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from multiagent.config import Settings  # noqa: E402
from multiagent.tools import SqlExecutionError, SqlSandbox, UnsafeSqlError, validate_sql  # noqa: E402

settings = Settings(_env_file=ROOT / ".env")


def _sandbox(**overrides) -> SqlSandbox:
    return SqlSandbox(Settings(_env_file=ROOT / ".env", **overrides))


def _require_db() -> SqlSandbox:
    box = _sandbox()
    try:
        box.describe_schema()
    except SqlExecutionError as exc:
        pytest.skip(f"数据库不可达（docker compose up -d 了吗？）: {exc}")
    return box


# ------------------------------------------------------- 语句校验（不需要数据库）


def test_accepts_plain_select() -> None:
    assert validate_sql("SELECT 1") == "SELECT 1"


def test_accepts_cte() -> None:
    sql = "WITH t AS (SELECT 1 AS a) SELECT * FROM t"
    assert validate_sql(sql) == sql


def test_accepts_trailing_semicolon() -> None:
    assert validate_sql("SELECT 1;") == "SELECT 1"


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO sales (revenue) VALUES (1)",
        "UPDATE sales SET revenue = 0",
        "DELETE FROM sales",
        "DROP TABLE sales",
        "ALTER TABLE sales ADD COLUMN x int",
        "CREATE TABLE x (a int)",
        "TRUNCATE sales",
        "GRANT ALL ON sales TO public",
        "COPY sales TO '/tmp/x.csv'",
        "SELECT 1; DROP TABLE sales",
    ],
)
def test_rejects_writing_statements(sql: str) -> None:
    with pytest.raises(UnsafeSqlError):
        validate_sql(sql)


def test_rejects_non_select_start() -> None:
    with pytest.raises(UnsafeSqlError, match="只允许"):
        validate_sql("EXPLAIN SELECT 1")


def test_rejects_empty() -> None:
    with pytest.raises(UnsafeSqlError):
        validate_sql("   ")


def test_allows_keyword_inside_string_literal() -> None:
    """字符串字面量里的关键字不应误伤，否则模型写 WHERE x='delete' 会被无故拒绝。"""
    sql = "SELECT * FROM sales WHERE channel = 'delete'"
    assert validate_sql(sql) == sql


def test_allows_keyword_as_substring_of_identifier() -> None:
    sql = "SELECT count(*) AS created_count FROM sales"
    assert validate_sql(sql) == sql


def test_ignores_line_comments() -> None:
    sql = "SELECT 1 -- 这里注释里写 drop table 不影响\n"
    assert "SELECT 1" in validate_sql(sql)


# ------------------------------------------------------- 真实执行（需要数据库）


def test_execute_returns_rows_with_traceable_hash() -> None:
    box = _require_db()
    result = box.execute(
        "SELECT region, SUM(revenue) AS revenue FROM sales "
        "WHERE stat_date = '2026-09-01' GROUP BY region ORDER BY region"
    )
    assert result.row_count == 2
    assert len(result.sql_hash) == 16, "sql_hash 用于结论溯源，必须稳定生成"
    assert {r["region"] for r in result.rows} == {"华东", "华南"}
    assert result.duration_ms >= 0


def test_execute_enforces_row_cap() -> None:
    box = _require_db()
    capped = _sandbox(sql_max_rows=3).execute("SELECT id FROM sales ORDER BY id")
    assert capped.row_count == 3
    assert capped.limited is True, "命中上限时应标记 limited，提示结果不完整"
    assert box.execute("SELECT id FROM sales").row_count == 24


def test_statement_timeout_actually_protects() -> None:
    """只读角色被设为 statement_timeout=5s，慢查询必须被数据库自己掐断。"""
    box = _require_db()
    with pytest.raises(SqlExecutionError, match="(?i)timeout|canceling"):
        box.execute("SELECT pg_sleep(8)")


def test_schema_introspection_lists_business_tables() -> None:
    box = _require_db()
    text = box.schema_as_prompt_text()
    assert "sales" in text
    assert "metric_definitions" in text
