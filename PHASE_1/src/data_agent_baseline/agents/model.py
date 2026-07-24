from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

from data_agent_baseline.events import EventSink, emit_event


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    id: str
    name: str
    arguments: str

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str | None
    tool_calls: tuple[ModelToolCall, ...] = field(default_factory=tuple)
    tool_call_id: str | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    tool_calls: tuple[ModelToolCall, ...]
    raw_response: str
    finish_reason: str | None = None


class ToolSchemaSource(Protocol):
    def to_openai_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError


class ModelAdapter(Protocol):
    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: ToolSchemaSource | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        raise NotImplementedError


def _serialize_response(
    *,
    content: str,
    tool_calls: tuple[ModelToolCall, ...],
    finish_reason: str | None,
) -> str:
    return json.dumps(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [call.to_openai_dict() for call in tool_calls],
            "finish_reason": finish_reason,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        request_timeout_seconds: float = 20.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 1.0,
        event_sink: EventSink | None = None,
        client: Any | None = None,
        sleep_fn=time.sleep,
        random_fn=random.random,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero.")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative.")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative.")

        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.event_sink = event_sink
        self._sleep = sleep_fn
        self._random = random_fn
        self._client = client
        if self._client is None and self.api_key:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.request_timeout_seconds,
                max_retries=0,
            )

    @staticmethod
    def _is_retryable(exc: APIError) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in {408, 409, 429} or exc.status_code >= 500
        return False

    @staticmethod
    def _retry_after_seconds(exc: APIError) -> float | None:
        if not isinstance(exc, APIStatusError):
            return None
        raw_value = exc.response.headers.get("retry-after")
        if not raw_value:
            return None
        try:
            return max(float(raw_value), 0.0)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw_value)
            except (TypeError, ValueError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)

    def _retry_delay_seconds(self, exc: APIError, retry_index: int) -> float:
        retry_after = self._retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, 5.0)
        exponential_delay = self.retry_backoff_seconds * (2**retry_index)
        jitter = self._random() * min(self.retry_backoff_seconds, 0.5)
        return min(exponential_delay + jitter, 5.0)

    @staticmethod
    def _parse_response(response: Any) -> ModelResponse:
        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")

        choice = choices[0]
        message = choice.message
        content_value = getattr(message, "content", None)
        content = content_value if isinstance(content_value, str) else ""
        parsed_calls: list[ModelToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            function = getattr(raw_call, "function", None)
            arguments = getattr(function, "arguments", "") if function is not None else ""
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            parsed_calls.append(
                ModelToolCall(
                    id=str(getattr(raw_call, "id", "") or ""),
                    name=str(getattr(function, "name", "") or "") if function is not None else "",
                    arguments=str(arguments or ""),
                )
            )

        tool_calls = tuple(parsed_calls)
        finish_reason_value = getattr(choice, "finish_reason", None)
        finish_reason = str(finish_reason_value) if finish_reason_value is not None else None
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=_serialize_response(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            ),
            finish_reason=finish_reason,
        )

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: ToolSchemaSource | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        if self._client is None:
            raise RuntimeError("Model client is not initialized.")

        context = dict(request_context or {})
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai_dict() for message in messages],
            "temperature": self.temperature,
        }
        if tools is not None:
            request_payload.update(
                {
                    "tools": tools.to_openai_tools(),
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            )

        max_attempts = self.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            started_at = time.perf_counter()
            emit_event(
                self.event_sink,
                "model_request_started",
                {
                    **context,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "timeout_seconds": self.request_timeout_seconds,
                },
            )
            try:
                raw_response = self._client.chat.completions.create(**request_payload)
            except APIError as exc:
                elapsed_seconds = round(time.perf_counter() - started_at, 3)
                retryable = self._is_retryable(exc)
                emit_event(
                    self.event_sink,
                    "model_request_failed",
                    {
                        **context,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "elapsed_seconds": elapsed_seconds,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retryable": retryable,
                    },
                )
                if not retryable or attempt >= max_attempts:
                    raise RuntimeError(
                        f"Model request failed after {attempt} attempt(s): {exc}"
                    ) from exc

                delay_seconds = self._retry_delay_seconds(exc, attempt - 1)
                emit_event(
                    self.event_sink,
                    "model_request_retry_scheduled",
                    {
                        **context,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "delay_seconds": round(delay_seconds, 3),
                    },
                )
                self._sleep(delay_seconds)
                continue

            response = self._parse_response(raw_response)
            emit_event(
                self.event_sink,
                "model_request_succeeded",
                {
                    **context,
                    "attempt": attempt,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                    "finish_reason": response.finish_reason,
                    "tool_call_count": len(response.tool_calls),
                },
            )
            return response

        raise AssertionError("Model retry loop exited unexpectedly.")


class ScriptedModelAdapter:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[list[ModelMessage]] = []

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: ToolSchemaSource | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        del tools
        del request_context
        self.requests.append(list(messages))
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
