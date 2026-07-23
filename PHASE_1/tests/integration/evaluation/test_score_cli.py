import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from data_agent_baseline.cli import app


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_score_run_command_writes_report(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    gold = tmp_path / "gold"
    inputs = tmp_path / "input"
    output = tmp_path / "report.json"
    predictions.mkdir()
    inputs.mkdir()

    _write_csv(gold / "task_1" / "gold.csv", [["value"], ["1"]])
    _write_csv(predictions / "task_1" / "prediction.csv", [["answer"], ["1"]])

    result = CliRunner().invoke(
        app,
        [
            "score-run",
            str(predictions),
            "--gold-dir",
            str(gold),
            "--input-dir",
            str(inputs),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Total score: 1.0000" in result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["aggregate"]["total_score"] == 1.0
    assert payload["per_task"]["task_1"]["status"] == "perfect_match"
