from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ListContextInput(ToolInput):
    max_depth: int = Field(
        default=4,
        ge=1,
        description="Maximum directory depth to include.",
    )


class ReadCsvInput(ToolInput):
    path: str = Field(description="Path relative to the task context directory.")
    max_rows: int = Field(
        default=20,
        ge=1,
        description="Maximum number of data rows to return.",
    )


class ReadJsonInput(ToolInput):
    path: str = Field(description="Path relative to the task context directory.")
    max_chars: int = Field(
        default=4000,
        ge=1,
        description="Maximum number of serialized characters to return.",
    )


class ReadDocInput(ToolInput):
    path: str = Field(description="Path relative to the task context directory.")
    max_chars: int = Field(
        default=4000,
        ge=1,
        description="Maximum number of document characters to return.",
    )


class InspectSqliteSchemaInput(ToolInput):
    path: str = Field(description="SQLite or DB path relative to the task context directory.")


class ExecuteContextSqlInput(ToolInput):
    path: str = Field(description="SQLite or DB path relative to the task context directory.")
    sql: str = Field(description="A read-only SELECT, WITH, or PRAGMA statement.")
    limit: int = Field(
        default=200,
        ge=1,
        description="Maximum number of rows to return.",
    )


class ExecutePythonInput(ToolInput):
    code: str = Field(description="Python source code to execute in the task context directory.")


class AnswerInput(ToolInput):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [
                {
                    "columns": ["average_long_shots"],
                    "rows": [["63.5"]],
                }
            ]
        },
    )

    columns: list[str] = Field(
        min_length=1,
        description=(
            "Names of ONLY the final result columns explicitly requested by the "
            "question. Do not include join keys, filter values, source columns, "
            "rankings, or intermediate statistics just for context: every extra "
            "column is penalized."
        ),
    )
    rows: list[list[Any]] = Field(
        description=(
            "Final-result rows in the same order as columns. Each row must contain "
            "exactly one value per submitted final column; a single-value question "
            "normally has one column and one row."
        )
    )

    @model_validator(mode="after")
    def validate_row_widths(self) -> AnswerInput:
        column_count = len(self.columns)
        if any(len(row) != column_count for row in self.rows):
            raise ValueError("each row must contain exactly one value per column")
        return self
