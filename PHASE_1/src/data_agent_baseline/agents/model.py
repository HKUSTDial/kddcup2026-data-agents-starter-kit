from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

from data_agent_baseline.events import EventSink, emit_event


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(
        self,
        messages: list[ModelMessage],
        *,
        request_context: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError


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

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        request_context: dict[str, Any] | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        if self._client is None:
            raise RuntimeError("Model client is not initialized.")

        context = dict(request_context or {})
        rendered_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
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
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=rendered_messages,
                    temperature=self.temperature,
                )
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

            choices = response.choices or []
            if not choices:
                raise RuntimeError("Model response missing choices.")
            content = choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("Model response missing text content.")
            emit_event(
                self.event_sink,
                "model_request_succeeded",
                {
                    **context,
                    "attempt": attempt,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                },
            )
            return content

        raise AssertionError("Model retry loop exited unexpectedly.")


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        request_context: dict[str, Any] | None = None,
    ) -> str:
        del messages
        del request_context
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
