"""Unit tests for ReActAgent using ScriptedModelAdapter.

ScriptedModelAdapter로 LLM 응답을 미리 주입하여 실제 API 호출 없이 에이전트 루프를 검증.

케이스:
- 단일 스텝에서 answer 제출 → 성공
- 중간 도구 사용 후 answer 제출 → 다중 스텝 성공
- 잘못된 JSON 응답 → error step 기록 후 계속
- max_steps 초과 → failure_reason 설정
- 응답 소진 → RuntimeError → failure_reason
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_task(context_dir: Path) -> PublicTask:
    record = TaskRecord(task_id="agent-test-001", difficulty="easy", question="List employees.")
    assets = TaskAssets(task_dir=context_dir.parent, context_dir=context_dir)
    return PublicTask(record=record, assets=assets)


def scripted_response(**kwargs) -> str:
    """ReAct JSON 응답 문자열 생성 헬퍼."""
    return json.dumps(kwargs)


def answer_response(columns: list[str], rows: list) -> str:
    return scripted_response(
        thought="I have the answer.",
        action="answer",
        action_input={"columns": columns, "rows": rows},
    )


@pytest.fixture()
def ctx(tmp_path: Path) -> Path:
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
# 단일 스텝 성공
# ---------------------------------------------------------------------------


class TestSingleStepSuccess:
    def test_answer_submitted_immediately(self, task: PublicTask, registry):
        model = ScriptedModelAdapter([
            answer_response(["name"], [["Alice"]])
        ])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=5))
        result = agent.run(task)

        assert result.answer is not None
        assert result.failure_reason is None
        assert result.answer.columns == ["name"]
        assert result.answer.rows == [["Alice"]]
        assert len(result.steps) == 1
        assert result.steps[0].action == "answer"
        assert result.steps[0].ok is True


# ---------------------------------------------------------------------------
# 다중 스텝 성공 (도구 사용 후 answer)
# ---------------------------------------------------------------------------


class TestMultiStepSuccess:
    def test_list_context_then_answer(self, task: PublicTask, ctx: Path, registry):
        (ctx / "data.csv").write_text("id,val\n1,x\n")

        model = ScriptedModelAdapter([
            scripted_response(
                thought="Let me see what files are available.",
                action="list_context",
                action_input={},
            ),
            answer_response(["result"], [["done"]]),
        ])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=5))
        result = agent.run(task)

        assert result.answer is not None
        assert result.failure_reason is None
        assert len(result.steps) == 2
        assert result.steps[0].action == "list_context"
        assert result.steps[1].action == "answer"

    def test_observation_from_tool_included_in_trace(self, task: PublicTask, ctx: Path, registry):
        (ctx / "info.txt").write_text("hello")

        model = ScriptedModelAdapter([
            scripted_response(
                thought="Read the doc.",
                action="read_doc",
                action_input={"path": "info.txt"},
            ),
            answer_response(["output"], [["hello"]]),
        ])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=5))
        result = agent.run(task)

        step0 = result.steps[0]
        assert step0.ok is True
        assert "hello" in str(step0.observation)


# ---------------------------------------------------------------------------
# 잘못된 JSON → error step
# ---------------------------------------------------------------------------


class TestInvalidJsonResponse:
    def test_bad_json_records_error_step_and_continues(self, task: PublicTask, registry):
        model = ScriptedModelAdapter([
            "this is not valid json at all",
            answer_response(["col"], [["val"]]),
        ])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=5))
        result = agent.run(task)

        # 첫 스텝은 에러, 두 번째 스텝은 answer
        assert len(result.steps) == 2
        assert result.steps[0].action == "__error__"
        assert result.steps[0].ok is False
        assert result.answer is not None

    def test_missing_action_field_records_error(self, task: PublicTask, registry):
        model = ScriptedModelAdapter([
            json.dumps({"thought": "thinking...", "action_input": {}}),  # action 누락
            answer_response(["x"], [[1]]),
        ])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=5))
        result = agent.run(task)

        assert result.steps[0].action == "__error__"
        assert result.answer is not None


# ---------------------------------------------------------------------------
# max_steps 초과
# ---------------------------------------------------------------------------


class TestMaxStepsExceeded:
    def test_failure_reason_set_when_no_answer(self, task: PublicTask, registry):
        # max_steps=2인데 list_context만 2번 실행하고 answer 없음
        model = ScriptedModelAdapter([
            scripted_response(thought=".", action="list_context", action_input={}),
            scripted_response(thought=".", action="list_context", action_input={}),
        ])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=2))
        result = agent.run(task)

        assert result.answer is None
        assert result.failure_reason is not None
        assert "max_steps" in result.failure_reason.lower() or "answer" in result.failure_reason.lower()

    def test_all_steps_recorded(self, task: PublicTask, registry):
        model = ScriptedModelAdapter([
            scripted_response(thought=".", action="list_context", action_input={}),
            scripted_response(thought=".", action="list_context", action_input={}),
            scripted_response(thought=".", action="list_context", action_input={}),
        ])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=3))
        result = agent.run(task)

        assert len(result.steps) == 3


# ---------------------------------------------------------------------------
# 응답 소진 (ScriptedModelAdapter 스크립트 부족)
# ---------------------------------------------------------------------------


class TestScriptedAdapterExhausted:
    def test_raises_runtime_error_propagates(self, task: PublicTask, registry):
        # model.complete()는 try/except 밖에서 호출되므로 RuntimeError가 그대로 전파됨
        model = ScriptedModelAdapter([])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=3))
        with pytest.raises(RuntimeError, match="No scripted model responses remaining"):
            agent.run(task)


# ---------------------------------------------------------------------------
# 알 수 없는 도구 → error step
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_unknown_action_records_error_step(self, task: PublicTask, registry):
        model = ScriptedModelAdapter([
            scripted_response(thought=".", action="fly_to_moon", action_input={}),
            answer_response(["x"], [["y"]]),
        ])
        agent = ReActAgent(model=model, tools=registry, config=ReActAgentConfig(max_steps=5))
        result = agent.run(task)

        assert result.steps[0].action == "__error__"
        assert result.answer is not None


# ---------------------------------------------------------------------------
# task_id 전달
# ---------------------------------------------------------------------------


class TestAgentRunResult:
    def test_task_id_propagated(self, task: PublicTask, registry):
        model = ScriptedModelAdapter([answer_response(["v"], [[1]])])
        agent = ReActAgent(model=model, tools=registry)
        result = agent.run(task)
        assert result.task_id == "agent-test-001"
