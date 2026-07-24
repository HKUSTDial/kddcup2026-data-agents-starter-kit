from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

from data_agent_baseline.agents.model import (
    ModelMessage,
    ModelToolCall,
    OpenAIModelAdapter,
)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.call_count = 0
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        self.call_count += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if hasattr(outcome, "choices"):
            return outcome
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=outcome, tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://example.test/chat"))


def _status_error(status_code: int, *, retry_after: str | None = None) -> APIStatusError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "https://example.test/chat")
    response = httpx.Response(status_code, request=request, headers=headers)
    return APIStatusError("request failed", response=response, body=None)


def _adapter(outcomes, *, max_retries=1, event_sink=None, sleep_fn=None):
    client = FakeClient(outcomes)
    sleeps = []
    adapter = OpenAIModelAdapter(
        model="test-model",
        api_base="https://example.test/v1",
        api_key="test-key",
        temperature=0.0,
        request_timeout_seconds=20.0,
        max_retries=max_retries,
        retry_backoff_seconds=1.0,
        event_sink=event_sink,
        client=client,
        sleep_fn=sleep_fn or sleeps.append,
        random_fn=lambda: 0.0,
    )
    return adapter, client, sleeps


class FakeTools:
    def to_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_context",
                    "description": "List files.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


def test_retries_transient_timeout_once_then_succeeds():
    events = []
    adapter, client, sleeps = _adapter(
        [_timeout_error(), "ok"],
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
    )

    result = adapter.complete(
        [ModelMessage(role="user", content="hello")],
        request_context={"task_id": "task_1", "step_index": 1},
    )

    assert result.content == "ok"
    assert client.completions.call_count == 2
    assert sleeps == [1.0]
    assert [event_type for event_type, _ in events] == [
        "model_request_started",
        "model_request_failed",
        "model_request_retry_scheduled",
        "model_request_started",
        "model_request_succeeded",
    ]


def test_stops_after_retry_budget_is_exhausted():
    adapter, client, sleeps = _adapter([_timeout_error(), _timeout_error()])

    with pytest.raises(RuntimeError, match=r"after 2 attempt\(s\)"):
        adapter.complete([ModelMessage(role="user", content="hello")])

    assert client.completions.call_count == 2
    assert sleeps == [1.0]


def test_does_not_retry_non_transient_client_error():
    adapter, client, sleeps = _adapter([_status_error(401)])

    with pytest.raises(RuntimeError, match=r"after 1 attempt\(s\)"):
        adapter.complete([ModelMessage(role="user", content="hello")])

    assert client.completions.call_count == 1
    assert sleeps == []


def test_caps_retry_after_header_at_five_seconds():
    adapter, client, sleeps = _adapter([_status_error(429, retry_after="30"), "ok"])

    assert adapter.complete([ModelMessage(role="user", content="hello")]).content == "ok"
    assert client.completions.call_count == 2
    assert sleeps == [5.0]


def test_sends_native_tool_schema_with_serial_execution():
    adapter, client, _ = _adapter(["ok"])

    adapter.complete(
        [ModelMessage(role="user", content="hello")],
        tools=FakeTools(),
    )

    request = client.completions.requests[0]
    assert request["tools"][0]["function"]["name"] == "list_context"
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is False


def test_parses_and_replays_native_tool_messages():
    raw_tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(name="list_context", arguments='{"max_depth":2}'),
    )
    raw_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[raw_tool_call]),
                finish_reason="tool_calls",
            )
        ]
    )
    adapter, client, _ = _adapter([raw_response])
    previous_call = ModelToolCall(
        id="call_previous",
        name="list_context",
        arguments='{"max_depth":1}',
    )

    response = adapter.complete(
        [
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=(previous_call,),
            ),
            ModelMessage(
                role="tool",
                content='{"ok":true}',
                tool_call_id="call_previous",
            ),
        ],
        tools=FakeTools(),
    )

    assert response.tool_calls == (
        ModelToolCall(
            id="call_123",
            name="list_context",
            arguments='{"max_depth":2}',
        ),
    )
    assert response.finish_reason == "tool_calls"
    request_messages = client.completions.requests[0]["messages"]
    assert request_messages[0]["tool_calls"][0]["id"] == "call_previous"
    assert request_messages[1]["tool_call_id"] == "call_previous"
