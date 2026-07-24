import json

from data_agent_baseline.agents.model import (
    ModelResponse,
    ModelToolCall,
    ScriptedModelAdapter,
)
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


def _task(tmp_path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "data.csv").write_text("value\none\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Return the value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _tool_response(
    name: str,
    arguments: dict[str, object] | str,
    *,
    call_id: str,
    content: str = "",
) -> ModelResponse:
    rendered_arguments = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    )
    call = ModelToolCall(id=call_id, name=name, arguments=rendered_arguments)
    return ModelResponse(
        content=content,
        tool_calls=(call,),
        raw_response=json.dumps({"tool_calls": [call.to_openai_dict()]}),
        finish_reason="tool_calls",
    )


def _answer_response(value: str = "one") -> ModelResponse:
    return _tool_response(
        "answer",
        {"columns": ["value"], "rows": [[value]]},
        call_id=f"call_answer_{value}",
    )


def test_runs_native_tool_loop_and_replays_matching_call_id(tmp_path):
    model = ScriptedModelAdapter(
        [
            _tool_response("list_context", {"max_depth": 2}, call_id="call_list"),
            _answer_response(),
        ]
    )
    events = []
    agent = ReActAgent(
        model=model,
        tools=create_default_tool_registry(),
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )

    result = agent.run(_task(tmp_path))

    assert result.succeeded is True
    assert [step.action for step in result.steps] == ["list_context", "answer"]
    assert result.steps[0].tool_call_id == "call_list"
    second_request = model.requests[1]
    assert [message.role for message in second_request] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_request[2].tool_calls[0].id == "call_list"
    assert second_request[3].tool_call_id == "call_list"
    tool_started = next(payload for kind, payload in events if kind == "tool_started")
    assert tool_started["tool_call_id"] == "call_list"


def test_returns_validation_error_to_model_and_allows_correction(tmp_path):
    model = ScriptedModelAdapter(
        [
            _tool_response("read_csv", {}, call_id="call_invalid"),
            _answer_response(),
        ]
    )
    events = []
    agent = ReActAgent(
        model=model,
        tools=create_default_tool_registry(),
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )

    result = agent.run(_task(tmp_path))

    assert result.succeeded is True
    assert result.steps[0].ok is False
    assert result.steps[0].observation["content"]["error"]["code"] == ("ARGUMENT_VALIDATION_ERROR")
    tool_message = model.requests[1][-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call_invalid"
    assert "ARGUMENT_VALIDATION_ERROR" in tool_message.content
    tool_failure = next(payload for kind, payload in events if kind == "tool_failed")
    assert tool_failure["error_code"] == "ARGUMENT_VALIDATION_ERROR"
    assert tool_failure["tool_call_id"] == "call_invalid"


def test_empty_tool_call_is_recorded_and_corrected_with_user_message(tmp_path):
    empty_response = ModelResponse(
        content="I will answer in text.",
        tool_calls=(),
        raw_response='{"content":"I will answer in text.","tool_calls":[]}',
        finish_reason="stop",
    )
    model = ScriptedModelAdapter([empty_response, _answer_response()])
    agent = ReActAgent(model=model, tools=create_default_tool_registry())

    result = agent.run(_task(tmp_path))

    assert result.succeeded is True
    assert result.steps[0].action == "__error__"
    assert result.steps[0].observation["content"]["error"]["code"] == "NO_TOOL_CALL"
    assert [message.role for message in model.requests[1][-2:]] == ["assistant", "user"]
    assert "native tool interface" in model.requests[1][-1].content


def test_multiple_tool_calls_are_rejected_with_one_tool_message_per_id(tmp_path):
    first_call = ModelToolCall(id="call_a", name="list_context", arguments="{}")
    second_call = ModelToolCall(id="call_b", name="list_context", arguments="{}")
    multiple_response = ModelResponse(
        content="",
        tool_calls=(first_call, second_call),
        raw_response='{"tool_calls":["call_a","call_b"]}',
        finish_reason="tool_calls",
    )
    model = ScriptedModelAdapter([multiple_response, _answer_response()])
    agent = ReActAgent(model=model, tools=create_default_tool_registry())

    result = agent.run(_task(tmp_path))

    assert result.succeeded is True
    assert result.steps[0].observation["content"]["error"]["code"] == ("MULTIPLE_TOOL_CALLS")
    replay = model.requests[1][-3:]
    assert [message.role for message in replay] == ["assistant", "tool", "tool"]
    assert [message.tool_call_id for message in replay[1:]] == ["call_a", "call_b"]


def test_max_steps_failure_preserves_protocol_error_trace(tmp_path):
    response = ModelResponse(
        content="No call.",
        tool_calls=(),
        raw_response='{"content":"No call.","tool_calls":[]}',
        finish_reason="stop",
    )
    agent = ReActAgent(
        model=ScriptedModelAdapter([response]),
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=1),
    )

    result = agent.run(_task(tmp_path))

    assert result.succeeded is False
    assert result.failure_reason == "Agent did not submit an answer within max_steps."
    assert result.steps[0].finish_reason == "stop"
