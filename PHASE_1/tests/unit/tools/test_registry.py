import json
import subprocess
import sys

from pydantic import BaseModel, ConfigDict

from data_agent_baseline.agents.model import ModelToolCall
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import (
    ToolRegistry,
    ToolSpec,
    create_default_tool_registry,
)


def test_registry_can_be_imported_in_a_fresh_interpreter():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from data_agent_baseline.tools.registry import create_default_tool_registry",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _task(tmp_path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Inspect the context.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _call(name: str, arguments: str) -> ModelToolCall:
    return ModelToolCall(id=f"call_{name}", name=name, arguments=arguments)


def test_default_registry_renders_openai_json_schemas():
    tools = create_default_tool_registry().to_openai_tools()

    assert [tool["function"]["name"] for tool in tools] == [
        "answer",
        "execute_context_sql",
        "execute_python",
        "inspect_sqlite_schema",
        "list_context",
        "read_csv",
        "read_doc",
        "read_json",
    ]
    for tool in tools:
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False

    read_csv = next(tool for tool in tools if tool["function"]["name"] == "read_csv")
    assert read_csv["function"]["parameters"]["required"] == ["path"]
    assert read_csv["function"]["parameters"]["properties"]["max_rows"]["default"] == 20

    answer = next(tool for tool in tools if tool["function"]["name"] == "answer")
    assert "ONLY the columns explicitly requested" in answer["function"]["description"]
    assert "Extra columns are penalized" in answer["function"]["description"]
    assert answer["function"]["parameters"]["examples"] == [
        {"columns": ["average_long_shots"], "rows": [["63.5"]]}
    ]
    assert "join keys" in answer["function"]["parameters"]["properties"]["columns"]["description"]


def test_rejects_invalid_json_wrong_types_and_extra_fields(tmp_path):
    task = _task(tmp_path)
    registry = create_default_tool_registry()

    invalid_json = registry.execute(task, _call("list_context", "{"))
    wrong_type = registry.execute(task, _call("list_context", '{"max_depth":"2"}'))
    extra_field = registry.execute(
        task,
        _call("list_context", '{"max_depth":2,"unexpected":true}'),
    )

    assert invalid_json.error_code == "INVALID_ARGUMENTS_JSON"
    assert wrong_type.error_code == "ARGUMENT_VALIDATION_ERROR"
    assert extra_field.error_code == "ARGUMENT_VALIDATION_ERROR"
    assert all(result.recoverable for result in (invalid_json, wrong_type, extra_field))


def test_rejects_unknown_tool_and_non_object_arguments(tmp_path):
    task = _task(tmp_path)
    registry = create_default_tool_registry()

    unknown = registry.execute(task, _call("missing_tool", "{}"))
    non_object = registry.execute(task, _call("list_context", "[]"))

    assert unknown.error_code == "UNKNOWN_TOOL"
    assert non_object.error_code == "ARGUMENTS_NOT_OBJECT"


def test_validates_answer_row_width_before_handler(tmp_path):
    task = _task(tmp_path)
    registry = create_default_tool_registry()
    arguments = json.dumps(
        {
            "columns": ["one", "two"],
            "rows": [["only-one-value"]],
        }
    )

    result = registry.execute(task, _call("answer", arguments))

    assert result.error_code == "ARGUMENT_VALIDATION_ERROR"
    assert result.is_terminal is False
    assert result.answer is None


def test_applies_defaults_and_returns_normalized_arguments(tmp_path):
    task = _task(tmp_path)
    result = create_default_tool_registry().execute(task, _call("list_context", "{}"))

    assert result.ok is True
    assert result.action_input == {"max_depth": 4}


class ExplodingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RequiredInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


def test_validation_failure_does_not_enter_handler(tmp_path):
    task = _task(tmp_path)
    calls = []

    def record_call(_task, arguments):
        calls.append(arguments)
        raise AssertionError("handler must not run")

    registry = ToolRegistry(
        specs={
            "required": ToolSpec(
                name="required",
                description="Requires an integer.",
                input_model=RequiredInput,
                handler=record_call,
            )
        }
    )

    result = registry.execute(task, _call("required", "{}"))

    assert result.error_code == "ARGUMENT_VALIDATION_ERROR"
    assert calls == []


def test_converts_handler_exceptions_to_recoverable_results(tmp_path):
    task = _task(tmp_path)

    def explode(_task, _arguments):
        raise OSError("synthetic failure")

    registry = ToolRegistry(
        specs={
            "explode": ToolSpec(
                name="explode",
                description="Always fails.",
                input_model=ExplodingInput,
                handler=explode,
            )
        }
    )

    result = registry.execute(task, _call("explode", "{}"))

    assert result.ok is False
    assert result.error_code == "TOOL_EXECUTION_ERROR"
    assert result.recoverable is True
    assert "OSError" in result.content["error"]["message"]
