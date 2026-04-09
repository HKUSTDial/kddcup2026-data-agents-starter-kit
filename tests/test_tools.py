"""Unit tests for individual tool handlers via ToolRegistry.execute().

Strategy:
- pytest tmp_path fixture으로 가짜 context 디렉토리 생성
- PublicTask를 직접 인스턴스화 (데이터셋 로더 우회)
- ToolRegistry.execute(task, action, action_input) 호출 후 ToolExecutionResult 검증
- 모델(LLM) 의존 없음 — 도구들은 순수 I/O
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import ToolExecutionResult, create_default_tool_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_task(context_dir: Path, task_id: str = "test-001") -> PublicTask:
    """PublicTask를 가짜 디렉토리로 직접 생성."""
    record = TaskRecord(task_id=task_id, difficulty="easy", question="What is the answer?")
    assets = TaskAssets(task_dir=context_dir.parent, context_dir=context_dir)
    return PublicTask(record=record, assets=assets)


@pytest.fixture()
def ctx(tmp_path: Path) -> Path:
    """비어있는 context 디렉토리."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    return context_dir


@pytest.fixture()
def task(ctx: Path) -> PublicTask:
    return make_task(ctx)


@pytest.fixture()
def registry():
    return create_default_tool_registry()


# ---------------------------------------------------------------------------
# list_context
# ---------------------------------------------------------------------------


class TestListContext:
    def test_empty_dir(self, task: PublicTask, registry):
        result = registry.execute(task, "list_context", {})
        assert result.ok
        assert result.content["entries"] == []

    def test_lists_files(self, task: PublicTask, ctx: Path, registry):
        (ctx / "data.csv").write_text("a,b\n1,2\n")
        (ctx / "notes.txt").write_text("hello")

        result = registry.execute(task, "list_context", {"max_depth": 1})
        names = {e["path"] for e in result.content["entries"]}
        assert "data.csv" in names
        assert "notes.txt" in names

    def test_subdirectory_traversal(self, task: PublicTask, ctx: Path, registry):
        sub = ctx / "sub"
        sub.mkdir()
        (sub / "nested.json").write_text("{}")

        result = registry.execute(task, "list_context", {"max_depth": 2})
        paths = {e["path"] for e in result.content["entries"]}
        assert "sub/nested.json" in paths

    def test_max_depth_limits_traversal(self, task: PublicTask, ctx: Path, registry):
        deep = ctx / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("x")

        result = registry.execute(task, "list_context", {"max_depth": 1})
        paths = {e["path"] for e in result.content["entries"]}
        assert "a/b/c/deep.txt" not in paths


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------


class TestReadCsv:
    def test_basic_preview(self, task: PublicTask, ctx: Path, registry):
        (ctx / "sales.csv").write_text("name,amount\nAlice,100\nBob,200\n")

        result = registry.execute(task, "read_csv", {"path": "sales.csv"})
        assert result.ok
        assert result.content["columns"] == ["name", "amount"]
        assert result.content["rows"] == [["Alice", "100"], ["Bob", "200"]]
        assert result.content["row_count"] == 2

    def test_max_rows_truncates(self, task: PublicTask, ctx: Path, registry):
        lines = ["id\n"] + [f"{i}\n" for i in range(50)]
        (ctx / "big.csv").write_text("".join(lines))

        result = registry.execute(task, "read_csv", {"path": "big.csv", "max_rows": 5})
        assert len(result.content["rows"]) == 5
        assert result.content["row_count"] == 50

    def test_empty_csv(self, task: PublicTask, ctx: Path, registry):
        (ctx / "empty.csv").write_text("")
        result = registry.execute(task, "read_csv", {"path": "empty.csv"})
        assert result.ok
        assert result.content["columns"] == []
        assert result.content["rows"] == []

    def test_missing_file_raises(self, task: PublicTask, registry):
        with pytest.raises(FileNotFoundError):
            registry.execute(task, "read_csv", {"path": "nonexistent.csv"})

    def test_path_traversal_blocked(self, task: PublicTask, registry):
        with pytest.raises(ValueError, match="escapes"):
            registry.execute(task, "read_csv", {"path": "../../../etc/passwd"})


# ---------------------------------------------------------------------------
# read_json
# ---------------------------------------------------------------------------


class TestReadJson:
    def test_basic(self, task: PublicTask, ctx: Path, registry):
        (ctx / "config.json").write_text(json.dumps({"key": "value"}))

        result = registry.execute(task, "read_json", {"path": "config.json"})
        assert result.ok
        assert '"key"' in result.content["preview"]
        assert not result.content["truncated"]

    def test_truncation(self, task: PublicTask, ctx: Path, registry):
        big_obj = {"data": "x" * 5000}
        (ctx / "big.json").write_text(json.dumps(big_obj))

        result = registry.execute(task, "read_json", {"path": "big.json", "max_chars": 100})
        assert result.content["truncated"]
        assert len(result.content["preview"]) == 100


# ---------------------------------------------------------------------------
# read_doc
# ---------------------------------------------------------------------------


class TestReadDoc:
    def test_text_file(self, task: PublicTask, ctx: Path, registry):
        (ctx / "readme.md").write_text("# Title\nSome content here.")

        result = registry.execute(task, "read_doc", {"path": "readme.md"})
        assert result.ok
        assert "Title" in result.content["preview"]
        assert not result.content["truncated"]

    def test_truncation(self, task: PublicTask, ctx: Path, registry):
        (ctx / "long.txt").write_text("a" * 5000)

        result = registry.execute(task, "read_doc", {"path": "long.txt", "max_chars": 200})
        assert result.content["truncated"]
        assert len(result.content["preview"]) == 200


# ---------------------------------------------------------------------------
# inspect_sqlite_schema
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_db(ctx: Path) -> Path:
    db_path = ctx / "store.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
    conn.execute("CREATE TABLE orders (order_id INTEGER, product_id INTEGER, qty INTEGER)")
    conn.commit()
    conn.close()
    return db_path


class TestInspectSqliteSchema:
    def test_lists_tables_and_columns(self, task: PublicTask, sqlite_db: Path, registry):
        result = registry.execute(task, "inspect_sqlite_schema", {"path": "store.db"})
        assert result.ok
        tables = {t["name"] for t in result.content["tables"]}
        assert "products" in tables
        assert "orders" in tables

    def test_column_names(self, task: PublicTask, sqlite_db: Path, registry):
        # inspect_sqlite_schema는 DDL을 create_sql 필드로 반환
        result = registry.execute(task, "inspect_sqlite_schema", {"path": "store.db"})
        products = next(t for t in result.content["tables"] if t["name"] == "products")
        create_sql = products["create_sql"]
        assert "id" in create_sql
        assert "name" in create_sql
        assert "price" in create_sql


# ---------------------------------------------------------------------------
# execute_context_sql
# ---------------------------------------------------------------------------


class TestExecuteContextSql:
    def test_select_returns_rows(self, task: PublicTask, sqlite_db: Path, registry):
        conn = sqlite3.connect(sqlite_db)
        conn.execute("INSERT INTO products VALUES (1, 'Widget', 9.99)")
        conn.commit()
        conn.close()

        result = registry.execute(
            task,
            "execute_context_sql",
            {"path": "store.db", "sql": "SELECT id, name FROM products"},
        )
        assert result.ok
        assert result.content["rows"] == [[1, "Widget"]]
        assert result.content["columns"] == ["id", "name"]

    def test_write_query_blocked(self, task: PublicTask, sqlite_db: Path, registry):
        # execute_read_only_sql은 SELECT/WITH/PRAGMA 외 쿼리에 ValueError를 raise
        with pytest.raises(ValueError, match="read-only"):
            registry.execute(
                task,
                "execute_context_sql",
                {"path": "store.db", "sql": "INSERT INTO products VALUES (99, 'X', 1.0)"},
            )

    def test_limit_applied(self, task: PublicTask, sqlite_db: Path, registry):
        conn = sqlite3.connect(sqlite_db)
        conn.executemany(
            "INSERT INTO products VALUES (?, ?, ?)",
            [(i, f"item{i}", float(i)) for i in range(1, 101)],
        )
        conn.commit()
        conn.close()

        result = registry.execute(
            task,
            "execute_context_sql",
            {"path": "store.db", "sql": "SELECT * FROM products", "limit": 10},
        )
        assert result.ok
        assert len(result.content["rows"]) == 10


# ---------------------------------------------------------------------------
# execute_python
# ---------------------------------------------------------------------------


class TestExecutePython:
    def test_stdout_captured(self, task: PublicTask, registry):
        result = registry.execute(
            task, "execute_python", {"code": "print('hello world')"}
        )
        assert result.ok
        assert "hello world" in result.content["output"]

    def test_can_read_context_files(self, task: PublicTask, ctx: Path, registry):
        (ctx / "data.txt").write_text("secret_value")
        code = "print(open('data.txt').read())"
        result = registry.execute(task, "execute_python", {"code": code})
        assert result.ok
        assert "secret_value" in result.content["output"]

    def test_syntax_error_returns_failure(self, task: PublicTask, registry):
        result = registry.execute(
            task, "execute_python", {"code": "def broken(: pass"}
        )
        assert not result.ok

    def test_runtime_exception_returns_failure(self, task: PublicTask, registry):
        result = registry.execute(
            task, "execute_python", {"code": "raise ValueError('oops')"}
        )
        assert not result.ok
        # 예외 메시지는 content["error"] 또는 content["traceback"]에 담김
        error_text = result.content.get("error", "") + result.content.get("traceback", "")
        assert "oops" in error_text

    def test_imports_available(self, task: PublicTask, registry):
        code = "import json; print(json.dumps({'x': 1}))"
        result = registry.execute(task, "execute_python", {"code": code})
        assert result.ok
        assert '"x"' in result.content["output"]


# ---------------------------------------------------------------------------
# answer (terminal tool)
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_basic_submission(self, task: PublicTask, registry):
        result = registry.execute(
            task,
            "answer",
            {"columns": ["name", "score"], "rows": [["Alice", 95], ["Bob", 87]]},
        )
        assert result.ok
        assert result.is_terminal
        assert result.answer is not None
        assert result.answer.columns == ["name", "score"]
        assert result.answer.rows == [["Alice", 95], ["Bob", 87]]

    def test_empty_rows_allowed(self, task: PublicTask, registry):
        result = registry.execute(
            task, "answer", {"columns": ["col"], "rows": []}
        )
        assert result.ok
        assert result.is_terminal
        assert result.answer.rows == []

    def test_missing_columns_raises(self, task: PublicTask, registry):
        with pytest.raises((ValueError, KeyError)):
            registry.execute(task, "answer", {"columns": [], "rows": []})

    def test_row_length_mismatch_raises(self, task: PublicTask, registry):
        with pytest.raises(ValueError):
            registry.execute(
                task, "answer", {"columns": ["a", "b"], "rows": [[1]]}
            )

    def test_non_string_columns_raises(self, task: PublicTask, registry):
        with pytest.raises((ValueError, TypeError)):
            registry.execute(
                task, "answer", {"columns": [1, 2], "rows": [[1, 2]]}
            )

    def test_content_reports_counts(self, task: PublicTask, registry):
        result = registry.execute(
            task,
            "answer",
            {"columns": ["x"], "rows": [[1], [2], [3]]},
        )
        assert result.content["column_count"] == 1
        assert result.content["row_count"] == 3

    def test_unknown_tool_raises(self, task: PublicTask, registry):
        with pytest.raises(KeyError):
            registry.execute(task, "nonexistent_tool", {})
