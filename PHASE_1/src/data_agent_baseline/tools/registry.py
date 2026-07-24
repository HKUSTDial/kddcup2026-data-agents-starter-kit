from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

from pydantic import BaseModel, ValidationError

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.contracts import (
    AnswerInput,
    ExecuteContextSqlInput,
    ExecutePythonInput,
    InspectSqliteSchemaInput,
    ListContextInput,
    ReadCsvInput,
    ReadDocInput,
    ReadJsonInput,
)
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    is_terminal: bool = False

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None
    action_input: dict[str, Any] | None = None
    error_code: str | None = None
    recoverable: bool = False


ToolHandler = Callable[[PublicTask, Any], ToolExecutionResult]


class ToolCall(Protocol):
    name: str
    arguments: str


def _list_context(task: PublicTask, action_input: ListContextInput) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=True,
        content=list_context_tree(task, max_depth=action_input.max_depth),
    )


def _read_csv(task: PublicTask, action_input: ReadCsvInput) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=True,
        content=read_csv_preview(task, action_input.path, max_rows=action_input.max_rows),
    )


def _read_json(task: PublicTask, action_input: ReadJsonInput) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=True,
        content=read_json_preview(task, action_input.path, max_chars=action_input.max_chars),
    )


def _read_doc(task: PublicTask, action_input: ReadDocInput) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(task, action_input.path, max_chars=action_input.max_chars),
    )


def _inspect_sqlite_schema(
    task: PublicTask,
    action_input: InspectSqliteSchemaInput,
) -> ToolExecutionResult:
    path = resolve_context_path(task, action_input.path)
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(
    task: PublicTask,
    action_input: ExecuteContextSqlInput,
) -> ToolExecutionResult:
    path = resolve_context_path(task, action_input.path)
    return ToolExecutionResult(
        ok=True,
        content=execute_read_only_sql(path, action_input.sql, limit=action_input.limit),
    )


def _execute_python(task: PublicTask, action_input: ExecutePythonInput) -> ToolExecutionResult:
    content = execute_python_code(
        context_root=task.context_dir,
        code=action_input.code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_: PublicTask, action_input: AnswerInput) -> ToolExecutionResult:
    normalized_rows = [list(row) for row in action_input.rows]
    answer = AnswerTable(columns=list(action_input.columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(action_input.columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


def _tool_error(
    *,
    code: str,
    message: str,
    action_input: dict[str, Any] | None = None,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=False,
        content={
            "error": {
                "code": code,
                "message": message,
                "recoverable": True,
            }
        },
        action_input=action_input,
        error_code=code,
        recoverable=True,
    )


def _validation_message(tool_name: str, error: ValidationError) -> str:
    details = []
    for item in error.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        details.append(f"{location}: {item['msg']}")
    return f"{tool_name} arguments are invalid: {'; '.join(details)}"


@dataclass(frozen=True, slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        rendered = []
        for name in sorted(self.specs):
            rendered.append(self.specs[name].to_openai_tool())
        return rendered

    def execute(self, task: PublicTask, call: ToolCall) -> ToolExecutionResult:
        spec = self.specs.get(call.name)
        if spec is None:
            return _tool_error(
                code="UNKNOWN_TOOL",
                message=f"Unknown tool: {call.name}",
            )

        try:
            raw_arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            return _tool_error(
                code="INVALID_ARGUMENTS_JSON",
                message=f"{call.name} arguments are not valid JSON: {exc.msg}",
            )
        if not isinstance(raw_arguments, dict):
            return _tool_error(
                code="ARGUMENTS_NOT_OBJECT",
                message=f"{call.name} arguments must decode to a JSON object.",
            )

        action_input = cast(dict[str, Any], raw_arguments)
        try:
            validated = spec.input_model.model_validate(action_input)
        except ValidationError as exc:
            return _tool_error(
                code="ARGUMENT_VALIDATION_ERROR",
                message=_validation_message(call.name, exc),
                action_input=action_input,
            )

        normalized_input = validated.model_dump(mode="json")
        try:
            result = spec.handler(task, validated)
        except Exception as exc:
            return _tool_error(
                code="TOOL_EXECUTION_ERROR",
                message=f"{call.name} failed: {type(exc).__name__}: {exc}",
                action_input=normalized_input,
            )
        return ToolExecutionResult(
            ok=result.ok,
            content=result.content,
            is_terminal=spec.is_terminal and result.ok and result.is_terminal,
            answer=result.answer,
            action_input=normalized_input,
            error_code=result.error_code,
            recoverable=result.recoverable,
        )


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description=(
                "Submit the final answer table and terminate the task. Submit ONLY "
                "the columns explicitly requested by the question; omit join keys, "
                "filter values, source columns, rankings, and intermediate "
                "statistics. Extra columns are penalized. Example: "
                '{"columns":["average_long_shots"],"rows":[["63.5"]]}.'
            ),
            input_model=AnswerInput,
            handler=_answer,
            is_terminal=True,
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_model=ExecuteContextSqlInput,
            handler=_execute_context_sql,
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_model=ExecutePythonInput,
            handler=_execute_python,
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_model=InspectSqliteSchemaInput,
            handler=_inspect_sqlite_schema,
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_model=ListContextInput,
            handler=_list_context,
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_model=ReadCsvInput,
            handler=_read_csv,
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            input_model=ReadDocInput,
            handler=_read_doc,
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_model=ReadJsonInput,
            handler=_read_json,
        ),
    }
    return ToolRegistry(specs=specs)
