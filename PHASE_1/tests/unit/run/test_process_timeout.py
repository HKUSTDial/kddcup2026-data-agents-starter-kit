import time
from pathlib import Path

from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.events import JsonlEventRecorder
from data_agent_baseline.run.runner import (
    _run_single_task_with_timeout,
    _write_json,
)


def _config(tmp_path: Path, *, timeout_seconds: float) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "data"),
        agent=AgentConfig(api_key="test-key"),
        run=RunConfig(
            output_dir=tmp_path / "runs",
            max_workers=1,
            task_timeout_seconds=timeout_seconds,
        ),
    )


def _slow_worker(task_id, config, result_path, events_path):
    del config
    del result_path
    recorder = JsonlEventRecorder(events_path, task_id=task_id)
    recorder.emit(
        "step_completed",
        {
            "step_index": 1,
            "step": {
                "step_index": 1,
                "thought": "partial",
                "action": "list_context",
                "action_input": {},
                "raw_response": "{}",
                "observation": {"ok": True},
                "ok": True,
            },
        },
    )
    recorder.emit(
        "model_request_started",
        {"step_index": 2, "attempt": 1, "max_attempts": 2},
    )
    time.sleep(10)


def _large_result_worker(task_id, config, result_path, events_path):
    del config
    del events_path
    _write_json(
        result_path,
        {
            "ok": True,
            "run_result": {
                "task_id": task_id,
                "answer": None,
                "steps": [],
                "failure_reason": "synthetic",
                "succeeded": False,
                "large_payload": "x" * 2_000_000,
            },
        },
    )


def test_timeout_preserves_completed_steps_and_current_request(tmp_path):
    task_output_dir = tmp_path / "run" / "task_1"
    task_output_dir.mkdir(parents=True)
    started_at = time.perf_counter()

    result = _run_single_task_with_timeout(
        task_id="task_1",
        config=_config(tmp_path, timeout_seconds=0.15),
        task_output_dir=task_output_dir,
        worker_target=_slow_worker,
    )

    assert time.perf_counter() - started_at < 2
    assert result["failure_reason"] == "Task timed out after 0.15 seconds."
    assert [step["step_index"] for step in result["steps"]] == [1]
    assert result["diagnostics"]["event_count"] == 3
    assert result["diagnostics"]["last_event_type"] == "task_timed_out"
    assert result["diagnostics"]["active_event_type"] == "model_request_started"
    assert result["diagnostics"]["last_step_index"] == 2
    assert result["diagnostics"]["last_attempt"] == 1
    assert (task_output_dir / "events.jsonl").is_file()


def test_large_worker_result_does_not_use_blocking_queue(tmp_path):
    task_output_dir = tmp_path / "run" / "task_1"
    task_output_dir.mkdir(parents=True)

    result = _run_single_task_with_timeout(
        task_id="task_1",
        config=_config(tmp_path, timeout_seconds=2.0),
        task_output_dir=task_output_dir,
        worker_target=_large_result_worker,
    )

    assert result["failure_reason"] == "synthetic"
    assert len(result["large_payload"]) == 2_000_000
    assert not (task_output_dir / "worker_result.json").exists()
