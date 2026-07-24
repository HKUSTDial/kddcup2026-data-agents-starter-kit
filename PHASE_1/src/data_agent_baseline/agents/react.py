from __future__ import annotations

import json
from dataclasses import dataclass

from data_agent_baseline.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.events import EventSink, emit_event
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


def _assistant_message(response: ModelResponse) -> ModelMessage:
    return ModelMessage(
        role="assistant",
        content=response.content,
        tool_calls=response.tool_calls,
    )


def _tool_message(call: ModelToolCall, observation: dict[str, object]) -> ModelMessage:
    return ModelMessage(
        role="tool",
        content=json.dumps(observation, ensure_ascii=False, separators=(",", ":"), default=str),
        tool_call_id=call.id,
    )


def _protocol_error_observation(code: str, message: str) -> dict[str, object]:
    return {
        "ok": False,
        "content": {
            "error": {
                "code": code,
                "message": message,
                "recoverable": True,
            }
        },
    }


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.event_sink = event_sink

    def _initial_messages(self, task: PublicTask) -> list[ModelMessage]:
        return [
            ModelMessage(
                role="system",
                content=build_system_prompt(system_prompt=self.system_prompt),
            ),
            ModelMessage(role="user", content=build_task_prompt(task)),
        ]

    def _record_protocol_error(
        self,
        *,
        state: AgentRuntimeState,
        messages: list[ModelMessage],
        response: ModelResponse,
        step_index: int,
        code: str,
        message: str,
    ) -> None:
        observation = _protocol_error_observation(code, message)
        emit_event(
            self.event_sink,
            "protocol_error",
            {
                "step_index": step_index,
                "error_code": code,
                "error": message,
                "tool_call_count": len(response.tool_calls),
            },
        )

        calls_are_replayable = bool(response.tool_calls) and all(
            call.id and call.name for call in response.tool_calls
        )
        if calls_are_replayable:
            messages.append(_assistant_message(response))
            for call in response.tool_calls:
                messages.append(_tool_message(call, observation))
        else:
            if response.content:
                messages.append(ModelMessage(role="assistant", content=response.content))
            messages.append(
                ModelMessage(
                    role="user",
                    content=(
                        f"Protocol error ({code}): {message} "
                        "Call exactly one provided tool through the native tool interface."
                    ),
                )
            )

        step_record = StepRecord(
            step_index=step_index,
            thought=response.content,
            action="__error__",
            action_input={},
            raw_response=response.raw_response,
            observation=observation,
            ok=False,
            finish_reason=response.finish_reason,
        )
        state.steps.append(step_record)
        emit_event(
            self.event_sink,
            "step_completed",
            {
                "step_index": step_index,
                "step": step_record.to_dict(),
            },
        )

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        messages = self._initial_messages(task)

        for step_index in range(1, self.config.max_steps + 1):
            emit_event(
                self.event_sink,
                "step_started",
                {"step_index": step_index},
            )
            response = self.model.complete(
                messages,
                tools=self.tools,
                request_context={
                    "task_id": task.task_id,
                    "step_index": step_index,
                },
            )

            if not response.tool_calls:
                self._record_protocol_error(
                    state=state,
                    messages=messages,
                    response=response,
                    step_index=step_index,
                    code="NO_TOOL_CALL",
                    message="The model response did not contain a native tool call.",
                )
                continue
            if len(response.tool_calls) != 1:
                self._record_protocol_error(
                    state=state,
                    messages=messages,
                    response=response,
                    step_index=step_index,
                    code="MULTIPLE_TOOL_CALLS",
                    message=(
                        "Parallel tool calls are disabled; the model must call exactly one "
                        "tool per turn."
                    ),
                )
                continue

            call = response.tool_calls[0]
            if not call.id or not call.name:
                self._record_protocol_error(
                    state=state,
                    messages=messages,
                    response=response,
                    step_index=step_index,
                    code="INVALID_TOOL_CALL",
                    message="A native tool call must include a non-empty id and function name.",
                )
                continue

            messages.append(_assistant_message(response))
            emit_event(
                self.event_sink,
                "tool_started",
                {
                    "step_index": step_index,
                    "tool_call_id": call.id,
                    "tool": call.name,
                    "arguments": call.arguments,
                },
            )
            tool_result = self.tools.execute(task, call)
            tool_event_payload = {
                "step_index": step_index,
                "tool_call_id": call.id,
                "tool": call.name,
                "ok": tool_result.ok,
                "is_terminal": tool_result.is_terminal,
                "error_code": tool_result.error_code,
                "recoverable": tool_result.recoverable,
            }
            if tool_result.error_code is not None:
                emit_event(
                    self.event_sink,
                    "tool_failed",
                    {
                        **tool_event_payload,
                        "error": tool_result.content,
                    },
                )
            else:
                emit_event(
                    self.event_sink,
                    "tool_completed",
                    tool_event_payload,
                )

            observation: dict[str, object] = {
                "ok": tool_result.ok,
                "tool": call.name,
                "content": tool_result.content,
            }
            messages.append(_tool_message(call, observation))
            step_record = StepRecord(
                step_index=step_index,
                thought=response.content,
                action=call.name,
                action_input=tool_result.action_input or {},
                raw_response=response.raw_response,
                observation=observation,
                ok=tool_result.ok,
                tool_call_id=call.id,
                finish_reason=response.finish_reason,
            )
            state.steps.append(step_record)
            emit_event(
                self.event_sink,
                "step_completed",
                {
                    "step_index": step_index,
                    "step": step_record.to_dict(),
                },
            )
            if tool_result.is_terminal:
                state.answer = tool_result.answer
                break

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
