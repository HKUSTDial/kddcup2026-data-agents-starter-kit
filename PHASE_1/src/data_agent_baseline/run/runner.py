from __future__ import annotations

import csv
import json
import multiprocessing
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.events import EventSink, JsonlEventRecorder, read_events
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    events_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path)
            if self.prediction_csv_path
            else None,
            "trace_path": str(self.trace_path),
            "events_path": str(self.events_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig, *, event_sink: EventSink | None = None):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        request_timeout_seconds=config.agent.model_request_timeout_seconds,
        max_retries=config.agent.model_max_retries,
        retry_backoff_seconds=config.agent.model_retry_backoff_seconds,
        event_sink=event_sink,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(
    task_id: str,
    failure_reason: str,
    *,
    steps: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "answer": None,
        "steps": steps or [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    event_sink: EventSink | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    agent = ReActAgent(
        model=model or build_model_adapter(config, event_sink=event_sink),
        tools=tools or create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=config.agent.max_steps),
        event_sink=event_sink,
    )
    run_result = agent.run(task)
    return run_result.to_dict()


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    result_path: Path,
    events_path: Path,
) -> None:
    recorder = JsonlEventRecorder(events_path, task_id=task_id)
    recorder.emit("task_started", {"task_timeout_seconds": config.run.task_timeout_seconds})
    try:
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            event_sink=recorder.emit,
        )
        recorder.emit(
            "task_completed",
            {
                "succeeded": bool(run_result.get("succeeded")),
                "failure_reason": run_result.get("failure_reason"),
            },
        )
        _write_json(result_path, {"ok": True, "run_result": run_result})
    except BaseException as exc:  # noqa: BLE001
        recorder.emit(
            "task_failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        _write_json(
            result_path,
            {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )


def _partial_steps_from_events(events_path: Path) -> list[dict[str, Any]]:
    steps_by_index: dict[int, dict[str, Any]] = {}
    for event in read_events(events_path):
        if event.get("event_type") != "step_completed":
            continue
        step = event.get("step")
        step_index = event.get("step_index")
        if isinstance(step, dict) and isinstance(step_index, int):
            steps_by_index[step_index] = step
    return [steps_by_index[index] for index in sorted(steps_by_index)]


def _last_event_diagnostics(events_path: Path) -> dict[str, Any]:
    events = read_events(events_path)
    if not events:
        return {"event_count": 0}
    last_event = events[-1]
    last_progress_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") not in {"task_timed_out", "task_failed"}
        ),
        last_event,
    )
    last_step_index = next(
        (
            event.get("step_index")
            for event in reversed(events)
            if isinstance(event.get("step_index"), int)
        ),
        None,
    )
    last_attempt = next(
        (
            event.get("attempt")
            for event in reversed(events)
            if isinstance(event.get("attempt"), int)
        ),
        None,
    )
    return {
        "event_count": len(events),
        "last_event_type": last_event.get("event_type"),
        "active_event_type": last_progress_event.get("event_type"),
        "active_tool": last_progress_event.get("tool"),
        "last_step_index": last_step_index,
        "last_attempt": last_attempt,
        "events_path": str(events_path),
    }


def _terminate_process(process: multiprocessing.Process) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join()


def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    task_output_dir: Path,
    worker_target: Callable[[str, AppConfig, Path, Path], None] = _run_single_task_in_subprocess,
) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    events_path = task_output_dir / "events.jsonl"
    result_path = task_output_dir / "worker_result.json"
    result_temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    result_path.unlink(missing_ok=True)
    result_temporary_path.unlink(missing_ok=True)
    if timeout_seconds <= 0:
        recorder = JsonlEventRecorder(events_path, task_id=task_id)
        return _run_single_task_core(
            task_id=task_id,
            config=config,
            event_sink=recorder.emit,
        )

    process = multiprocessing.Process(
        target=worker_target,
        args=(task_id, config, result_path, events_path),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        _terminate_process(process)
        result_temporary_path.unlink(missing_ok=True)
        recorder = JsonlEventRecorder(events_path, task_id=task_id)
        recorder.emit(
            "task_timed_out",
            {
                "timeout_seconds": timeout_seconds,
            },
        )
        return _failure_run_result_payload(
            task_id,
            f"Task timed out after {timeout_seconds:g} seconds.",
            steps=_partial_steps_from_events(events_path),
            diagnostics=_last_event_diagnostics(events_path),
        )

    if not result_path.is_file():
        result_temporary_path.unlink(missing_ok=True)
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
                steps=_partial_steps_from_events(events_path),
                diagnostics=_last_event_diagnostics(events_path),
            )
        return _failure_run_result_payload(
            task_id,
            "Task exited without returning a result.",
            steps=_partial_steps_from_events(events_path),
            diagnostics=_last_event_diagnostics(events_path),
        )

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return _failure_run_result_payload(
            task_id,
            f"Task returned an unreadable worker result: {exc}",
            steps=_partial_steps_from_events(events_path),
            diagnostics=_last_event_diagnostics(events_path),
        )
    finally:
        result_path.unlink(missing_ok=True)

    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(
        task_id,
        f"Task failed with uncaught error: {result['error']}",
        steps=_partial_steps_from_events(events_path),
        diagnostics=_last_event_diagnostics(events_path),
    )


def _write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any]
) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    events_path = task_output_dir / "events.jsonl"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        events_path=events_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    events_path = task_output_dir / "events.jsonl"
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(
            task_id=task_id,
            config=config,
            task_output_dir=task_output_dir,
        )
    else:
        recorder = JsonlEventRecorder(events_path, task_id=task_id)
        recorder.emit("task_started", {"task_timeout_seconds": None})
        try:
            run_result = _run_single_task_core(
                task_id=task_id,
                config=config,
                model=model,
                tools=tools,
                event_sink=recorder.emit,
            )
            recorder.emit(
                "task_completed",
                {
                    "succeeded": bool(run_result.get("succeeded")),
                    "failure_reason": run_result.get("failure_reason"),
                },
            )
        except BaseException as exc:  # noqa: BLE001
            recorder.emit(
                "task_failed",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            run_result = _failure_run_result_payload(
                task_id,
                f"Task failed with uncaught error: {exc}",
                steps=_partial_steps_from_events(events_path),
                diagnostics=_last_event_diagnostics(events_path),
            )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def _load_task_artifacts(run_output_dir: Path, task_id: str) -> TaskRunArtifacts | None:
    task_output_dir = run_output_dir / task_id
    trace_path = task_output_dir / "trace.json"
    if not trace_path.is_file():
        return None
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(trace, dict) or not isinstance(trace.get("succeeded"), bool):
        return None

    prediction_candidate = task_output_dir / "prediction.csv"
    succeeded = bool(trace["succeeded"])
    if succeeded and not prediction_candidate.is_file():
        return None
    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_candidate if prediction_candidate.is_file() else None,
        trace_path=trace_path,
        events_path=task_output_dir / "events.jsonl",
        succeeded=succeeded,
        failure_reason=trace.get("failure_reason"),
    )


def _archive_task_attempt(task_output_dir: Path) -> None:
    artifact_names = (
        "trace.json",
        "prediction.csv",
        "events.jsonl",
        "worker_result.json",
        ".worker_result.json.tmp",
    )
    existing_paths = [
        task_output_dir / artifact_name
        for artifact_name in artifact_names
        if (task_output_dir / artifact_name).exists()
    ]
    if not existing_paths:
        return

    attempts_dir = task_output_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = 1
    while (attempts_dir / f"attempt_{attempt_number:03d}").exists():
        attempt_number += 1
    archive_dir = attempts_dir / f"attempt_{attempt_number:03d}"
    archive_dir.mkdir()
    for source_path in existing_paths:
        shutil.move(str(source_path), archive_dir / source_path.name)


def _read_run_manifest(run_output_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_output_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    summary_path = run_output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    summary_tasks = summary.get("tasks", []) if isinstance(summary, dict) else []
    task_ids = [
        task["task_id"]
        for task in summary_tasks
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    ]
    if not task_ids:
        return None
    return {
        "schema_version": 1,
        "run_id": run_output_dir.name,
        "task_ids": task_ids,
        "created_at": None,
        "recovered_from_summary": True,
    }


def _validate_task_ids(dataset: DABenchPublicDataset, task_ids: list[str]) -> list[str]:
    if not task_ids:
        raise ValueError("At least one task must be selected.")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Selected task IDs must not contain duplicates.")

    available_task_ids = set(dataset.list_task_ids())
    unknown_task_ids = [task_id for task_id in task_ids if task_id not in available_task_ids]
    if unknown_task_ids:
        raise ValueError(f"Unknown task IDs: {', '.join(unknown_task_ids)}")
    return list(task_ids)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    if retry_failed and not resume:
        raise ValueError("retry_failed requires resume=True.")
    if task_ids is not None and limit is not None:
        raise ValueError("task_ids and limit cannot be used together.")
    dataset = DABenchPublicDataset(config.dataset.root_path)
    requested_task_ids = (
        _validate_task_ids(dataset, task_ids)
        if task_ids is not None
        else [task.task_id for task in dataset.iter_tasks()]
    )
    if limit is not None:
        requested_task_ids = requested_task_ids[:limit]

    if resume:
        if config.run.run_id is None:
            raise ValueError("run.run_id is required when resuming a run.")
        effective_run_id = resolve_run_id(config.run.run_id)
        run_output_dir = config.run.output_dir / effective_run_id
        if not run_output_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_output_dir}")
        manifest = _read_run_manifest(run_output_dir)
        if manifest is None:
            raise ValueError("Run manifest is missing or unreadable.")
        manifest_task_ids = manifest.get("task_ids")
        if not isinstance(manifest_task_ids, list) or not all(
            isinstance(task_id, str) for task_id in manifest_task_ids
        ):
            raise ValueError("Run manifest has invalid task_ids.")
        manifest_task_ids = _validate_task_ids(dataset, list(manifest_task_ids))
        if task_ids is None and limit is None:
            requested_task_ids = manifest_task_ids
        elif requested_task_ids != manifest_task_ids:
            raise ValueError("Resume selection must match the original run manifest.")
    else:
        effective_run_id, run_output_dir = create_run_output_dir(
            config.run.output_dir,
            run_id=config.run.run_id,
        )
        _write_json(
            run_output_dir / "manifest.json",
            {
                "schema_version": 1,
                "run_id": effective_run_id,
                "task_ids": requested_task_ids,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(requested_task_ids)
    task_ids_to_run: list[str] = []
    task_index_by_id = {task_id: index for index, task_id in enumerate(requested_task_ids)}
    resumed_task_count = 0
    for task_id in requested_task_ids:
        existing_artifact = _load_task_artifacts(run_output_dir, task_id) if resume else None
        should_retry = existing_artifact is not None and not existing_artifact.succeeded
        if existing_artifact is not None and not (retry_failed and should_retry):
            indexed_artifacts[task_index_by_id[task_id]] = existing_artifact
            resumed_task_count += 1
            if progress_callback is not None:
                progress_callback(existing_artifact)
            continue

        task_output_dir = run_output_dir / task_id
        if resume:
            _archive_task_attempt(task_output_dir)
        task_ids_to_run.append(task_id)

    if effective_workers == 1:
        for task_id in task_ids_to_run:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=model,
                tools=tools,
            )
            indexed_artifacts[task_index_by_id[task_id]] = artifact
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_task_id = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): task_id
                for task_id in task_ids_to_run
            }
            for future in as_completed(future_to_task_id):
                artifact = future.result()
                task_id = future_to_task_id[future]
                indexed_artifacts[task_index_by_id[task_id]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)

    task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]
    if len(task_artifacts) != len(requested_task_ids):
        raise RuntimeError("Benchmark finished without artifacts for every selected task.")

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "resumed": resume,
            "resumed_task_count": resumed_task_count,
            "executed_task_count": len(task_ids_to_run),
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
