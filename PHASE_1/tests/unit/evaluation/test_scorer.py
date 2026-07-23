import csv
import json
from pathlib import Path

import pytest

from data_agent_baseline.evaluation.scorer import (
    EvaluationError,
    evaluate_predictions,
    score_task,
)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_exact_content_matches_despite_names_and_order(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    _write_csv(gold, [["id", "sex"], ["1", "F"], ["2", "M"]])
    _write_csv(prediction, [["renamed_sex", "renamed_id"], ["M", "2"], ["F", "1"]])

    result = score_task(prediction, gold)

    assert result.status == "perfect_match"
    assert result.score == 1.0
    assert result.matched_columns == 2


def test_extra_column_receives_soft_penalty(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    _write_csv(gold, [["b", "c"], ["1", "x"], ["2", "y"]])
    _write_csv(
        prediction,
        [["b", "c", "extra"], ["1", "x", "noise-1"], ["2", "y", "noise-2"]],
    )

    result = score_task(prediction, gold, penalty_weight=0.1)

    assert result.recall == 1.0
    assert result.extra_columns == 1
    assert result.penalty == pytest.approx(0.1 / 3)
    assert result.score == pytest.approx(1 - 0.1 / 3)


def test_missing_gold_column_reduces_recall_without_extra_penalty(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    _write_csv(gold, [["b", "c"], ["1", "x"], ["2", "y"]])
    _write_csv(prediction, [["b"], ["1"], ["2"]])

    result = score_task(prediction, gold)

    assert result.status == "partial_match"
    assert result.recall == 0.5
    assert result.penalty == 0.0
    assert result.score == 0.5


def test_wrong_predicted_column_is_both_missing_and_extra(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    _write_csv(gold, [["b", "c"], ["1", "x"], ["2", "y"]])
    _write_csv(prediction, [["b", "wrong"], ["1", "q"], ["2", "z"]])

    result = score_task(prediction, gold)

    assert result.recall == 0.5
    assert result.extra_columns == 1
    assert result.penalty == pytest.approx(0.05)
    assert result.score == pytest.approx(0.45)


def test_extra_row_invalidates_the_affected_column(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    _write_csv(gold, [["value"], ["1"], ["2"]])
    _write_csv(prediction, [["value"], ["1"], ["2"], ["3"]])

    result = score_task(prediction, gold)

    assert result.status == "no_match"
    assert result.score == 0.0


def test_cross_column_row_relationship_is_not_scored(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    _write_csv(gold, [["id", "sex"], ["1", "F"], ["2", "M"]])
    _write_csv(prediction, [["id", "sex"], ["1", "M"], ["2", "F"]])

    result = score_task(prediction, gold)

    assert result.status == "perfect_match"
    assert result.score == 1.0


def test_duplicate_column_signatures_match_one_to_one(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    _write_csv(gold, [["first", "second"], ["1", "1"], ["2", "2"]])
    _write_csv(prediction, [["only"], ["1"], ["2"]])

    result = score_task(prediction, gold)

    assert result.matched_columns == 1
    assert result.gold_columns == 2
    assert result.score == 0.5


def test_missing_prediction_scores_zero(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    _write_csv(gold, [["value"], ["1"]])

    result = score_task(tmp_path / "missing.csv", gold)

    assert result.status == "missing_prediction"
    assert result.score == 0.0


def test_malformed_prediction_scores_zero_with_diagnostic(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    _write_csv(gold, [["a", "b"], ["1", "2"]])
    prediction.write_text("a,b\n1\n", encoding="utf-8")

    result = score_task(prediction, gold)

    assert result.status == "read_error:ValueError"
    assert result.score == 0.0
    assert "expected 2" in (result.error or "")


def test_invalid_gold_fails_evaluation(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    prediction = tmp_path / "prediction.csv"
    gold.write_text("", encoding="utf-8")
    _write_csv(prediction, [["value"], ["1"]])

    with pytest.raises(EvaluationError, match="Invalid gold CSV"):
        score_task(prediction, gold)


def test_aggregate_includes_tasks_without_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    gold = tmp_path / "gold"
    inputs = tmp_path / "input"
    predictions.mkdir()

    _write_csv(gold / "task_1" / "gold.csv", [["value"], ["1"]])
    _write_csv(gold / "task_2" / "gold.csv", [["value"], ["2"]])
    _write_csv(predictions / "task_1" / "prediction.csv", [["answer"], ["1"]])
    for task_id, difficulty in (("task_1", "easy"), ("task_2", "hard")):
        task_dir = inputs / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "question": "fixture",
                }
            ),
            encoding="utf-8",
        )

    report = evaluate_predictions(predictions, gold, input_root=inputs)

    assert list(report.per_task) == ["task_1", "task_2"]
    assert report.aggregate["total_score"] == 0.5
    assert report.aggregate["tasks_missing"] == 1
    assert report.aggregate["by_difficulty"] == {
        "easy": {"count": 1, "mean_score": 1.0},
        "hard": {"count": 1, "mean_score": 0.0},
    }
