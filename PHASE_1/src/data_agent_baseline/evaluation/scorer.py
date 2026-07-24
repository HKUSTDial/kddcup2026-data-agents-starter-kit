from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_agent_baseline.evaluation.normalization import column_signature


class EvaluationError(RuntimeError):
    """Raised when the trusted evaluation inputs or configuration are invalid."""


@dataclass(frozen=True, slots=True)
class TaskScore:
    status: str
    score: float = 0.0
    recall: float = 0.0
    penalty: float = 0.0
    gold_columns: int = 0
    predicted_columns: int = 0
    matched_columns: int = 0
    extra_columns: int = 0
    missing_gold_signatures: list[str] = field(default_factory=list)
    extra_prediction_signatures: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    meta: dict[str, Any]
    per_task: dict[str, TaskScore]
    aggregate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "per_task": {
                task_id: task_score.to_dict() for task_id, task_score in self.per_task.items()
            },
            "aggregate": dict(self.aggregate),
        }


def _read_csv_columns(path: Path) -> list[list[str]]:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise EvaluationError(f"Cannot open CSV {path}: {exc}") from exc

    with handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV has no header: {path}") from exc

        if not header:
            raise ValueError(f"CSV has no columns: {path}")

        columns: list[list[str]] = [[] for _ in header]
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"CSV row {row_number} has {len(row)} values; expected {len(header)}."
                )
            for column_index, value in enumerate(row):
                columns[column_index].append(value)
    return columns


def _signature_counts(
    columns: list[list[str]],
    *,
    strict_commas: bool,
) -> Counter[str]:
    return Counter(column_signature(column, strict_commas=strict_commas) for column in columns)


def score_task(
    prediction_csv: Path,
    gold_csv: Path,
    *,
    penalty_weight: float = 0.1,
    strict_commas: bool = False,
) -> TaskScore:
    """Score one prediction using official column-signature matching."""

    if penalty_weight < 0:
        raise ValueError("penalty_weight must be non-negative.")
    if not gold_csv.is_file():
        raise EvaluationError(f"Missing gold CSV: {gold_csv}")
    if not prediction_csv.is_file():
        return TaskScore(status="missing_prediction")

    try:
        gold_columns = _read_csv_columns(gold_csv)
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationError(f"Invalid gold CSV {gold_csv}: {exc}") from exc

    try:
        prediction_columns = _read_csv_columns(prediction_csv)
    except Exception as exc:
        return TaskScore(
            status=f"read_error:{type(exc).__name__}",
            error=str(exc),
        )

    gold_signatures = _signature_counts(gold_columns, strict_commas=strict_commas)
    prediction_signatures = _signature_counts(
        prediction_columns,
        strict_commas=strict_commas,
    )

    matched_columns = sum(
        min(gold_count, prediction_signatures.get(signature, 0))
        for signature, gold_count in gold_signatures.items()
    )
    gold_column_count = sum(gold_signatures.values())
    prediction_column_count = sum(prediction_signatures.values())
    extra_columns = max(0, prediction_column_count - matched_columns)

    recall = matched_columns / gold_column_count if gold_column_count else 0.0
    penalty = (
        penalty_weight * (extra_columns / prediction_column_count)
        if prediction_column_count
        else 0.0
    )
    score = max(0.0, recall - penalty)

    missing_gold_signatures: list[str] = []
    for signature, gold_count in gold_signatures.items():
        missing_count = gold_count - prediction_signatures.get(signature, 0)
        if missing_count > 0:
            missing_gold_signatures.extend([signature] * missing_count)

    extra_prediction_signatures: list[str] = []
    for signature, prediction_count in prediction_signatures.items():
        extra_count = prediction_count - gold_signatures.get(signature, 0)
        if extra_count > 0:
            extra_prediction_signatures.extend([signature] * extra_count)

    if recall == 1.0 and extra_columns == 0:
        status = "perfect_match"
    elif matched_columns:
        status = "partial_match"
    else:
        status = "no_match"

    return TaskScore(
        status=status,
        score=score,
        recall=recall,
        penalty=penalty,
        gold_columns=gold_column_count,
        predicted_columns=prediction_column_count,
        matched_columns=matched_columns,
        extra_columns=extra_columns,
        missing_gold_signatures=missing_gold_signatures,
        extra_prediction_signatures=extra_prediction_signatures,
    )


def _task_sort_key(task_id: str) -> tuple[int, int | str]:
    prefix = "task_"
    if task_id.startswith(prefix):
        suffix = task_id.removeprefix(prefix)
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, task_id)


def _read_difficulty(input_root: Path | None, task_id: str) -> str:
    if input_root is None:
        return "unknown"
    metadata_path = input_root / task_id / "task.json"
    if not metadata_path.is_file():
        return "unknown"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return str(payload.get("difficulty", "unknown"))


def _aggregate(
    per_task: dict[str, TaskScore],
    difficulties: dict[str, str],
) -> dict[str, Any]:
    if not per_task:
        return {
            "total_score": 0.0,
            "mean_recall": 0.0,
            "task_count": 0,
            "tasks_scored_nonzero": 0,
            "tasks_missing": 0,
            "tasks_with_read_errors": 0,
            "by_difficulty": {},
        }

    scores_by_difficulty: dict[str, list[float]] = defaultdict(list)
    for task_id, task_score in per_task.items():
        scores_by_difficulty[difficulties.get(task_id, "unknown")].append(task_score.score)

    task_count = len(per_task)
    return {
        "total_score": sum(item.score for item in per_task.values()) / task_count,
        "mean_recall": sum(item.recall for item in per_task.values()) / task_count,
        "task_count": task_count,
        "tasks_scored_nonzero": sum(1 for item in per_task.values() if item.score > 0),
        "tasks_missing": sum(
            1 for item in per_task.values() if item.status == "missing_prediction"
        ),
        "tasks_with_read_errors": sum(
            1 for item in per_task.values() if item.status.startswith("read_error:")
        ),
        "by_difficulty": {
            difficulty: {
                "count": len(scores),
                "mean_score": sum(scores) / len(scores),
            }
            for difficulty, scores in sorted(scores_by_difficulty.items())
        },
    }


def evaluate_predictions(
    predictions_root: Path,
    gold_root: Path,
    *,
    input_root: Path | None = None,
    penalty_weight: float = 0.1,
    strict_commas: bool = False,
) -> EvaluationReport:
    """Evaluate every task containing a gold.csv under ``gold_root``."""

    if penalty_weight < 0:
        raise ValueError("penalty_weight must be non-negative.")
    if not predictions_root.is_dir():
        raise EvaluationError(f"Missing predictions directory: {predictions_root}")
    if not gold_root.is_dir():
        raise EvaluationError(f"Missing gold directory: {gold_root}")

    task_ids = sorted(
        {
            gold_path.parent.name
            for gold_path in gold_root.glob("task_*/gold.csv")
            if gold_path.is_file()
        },
        key=_task_sort_key,
    )
    if not task_ids:
        raise EvaluationError(f"No task_*/gold.csv files found under {gold_root}")

    per_task: dict[str, TaskScore] = {}
    difficulties: dict[str, str] = {}
    for task_id in task_ids:
        per_task[task_id] = score_task(
            predictions_root / task_id / "prediction.csv",
            gold_root / task_id / "gold.csv",
            penalty_weight=penalty_weight,
            strict_commas=strict_commas,
        )
        difficulties[task_id] = _read_difficulty(input_root, task_id)

    return EvaluationReport(
        meta={
            "predictions_root": str(predictions_root),
            "gold_root": str(gold_root),
            "input_root": str(input_root) if input_root is not None else None,
            "penalty_weight": penalty_weight,
            "strict_commas": strict_commas,
            "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        per_task=per_task,
        aggregate=_aggregate(per_task, difficulties),
    )


def write_evaluation_report(report: EvaluationReport, output_path: Path) -> None:
    """Atomically write a score report as UTF-8 JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
