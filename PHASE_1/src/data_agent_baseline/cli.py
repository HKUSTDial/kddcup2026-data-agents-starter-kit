import json
from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.evaluation import (
    EvaluationError,
    evaluate_predictions,
    write_evaluation_report,
)
from data_agent_baseline.run.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark,
    run_single_task,
)
from data_agent_baseline.tools.filesystem import list_context_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _load_task_file(path: Path) -> list[str]:
    task_ids = [
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]
    if not task_ids:
        raise typer.BadParameter(
            "Task file must contain at least one task ID.", param_hint="--task-file"
        )
    if len(task_ids) != len(set(task_ids)):
        raise typer.BadParameter("Task file contains duplicate task IDs.", param_hint="--task-file")
    return task_ids


def _resume_task_total(app_config) -> int | None:
    run_id = app_config.run.run_id
    if run_id is None:
        return None
    manifest_path = app_config.run.output_dir / run_id / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    task_ids = manifest.get("task_ids") if isinstance(manifest, dict) else None
    if not isinstance(task_ids, list) or not all(isinstance(task_id, str) for task_id in task_ids):
        return None
    return len(task_ids)


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row(
        "dataset_root",
        str(app_config.dataset.root_path),
        _status_value(app_config.dataset.root_path),
    )
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("score-run")
def score_run_command(
    predictions_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Run directory containing task_<id>/prediction.csv files.",
    ),
    gold_dir: Path = typer.Option(
        PROJECT_ROOT / "data" / "public" / "output",
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Directory containing task_<id>/gold.csv files.",
    ),
    input_dir: Path | None = typer.Option(
        PROJECT_ROOT / "data" / "public" / "input",
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Optional task input directory used for difficulty metadata.",
    ),
    output: Path | None = typer.Option(
        None,
        dir_okay=False,
        resolve_path=True,
        help="Score report path. Defaults to <predictions_dir>/scores.json.",
    ),
    penalty_weight: float = typer.Option(
        0.1,
        min=0.0,
        help="Penalty weight applied to unmatched extra columns.",
    ),
    strict_commas: bool = typer.Option(
        False,
        help="Keep numeric grouping commas instead of removing them.",
    ),
    verbose: bool = typer.Option(
        False,
        help="Print one score row per task.",
    ),
) -> None:
    """Score a prediction run against public gold answers."""

    output_path = output or (predictions_dir / "scores.json")
    try:
        report = evaluate_predictions(
            predictions_dir,
            gold_dir,
            input_root=input_dir,
            penalty_weight=penalty_weight,
            strict_commas=strict_commas,
        )
        write_evaluation_report(report, output_path)
    except (EvaluationError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    aggregate = report.aggregate
    console.print(f"Score report: {output_path}")
    console.print(
        "Total score: "
        f"{aggregate['total_score']:.4f} "
        f"({aggregate['tasks_scored_nonzero']}/{aggregate['task_count']} tasks non-zero; "
        f"{aggregate['tasks_missing']} missing)"
    )

    if verbose:
        table = Table(title="Task Scores")
        table.add_column("Task")
        table.add_column("Status")
        table.add_column("Recall", justify="right")
        table.add_column("Penalty", justify="right")
        table.add_column("Score", justify="right")
        for task_id, task_score in report.per_task.items():
            table.add_row(
                task_id,
                task_score.status,
                f"{task_score.recall:.3f}",
                f"{task_score.penalty:.3f}",
                f"{task_score.score:.3f}",
            )
        console.print(table)


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the ReAct baseline on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir, run_id=app_config.run.run_id
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    console.print(f"Events JSONL: {artifacts.events_path}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    task_file: Path | None = typer.Option(
        None,
        "--task-file",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="Text file containing one task ID per line.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume the run identified by run.run_id.",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="When resuming, archive and rerun completed failures.",
    ),
) -> None:
    """Run the ReAct baseline on multiple tasks from the config selection."""
    if limit is not None and task_file is not None:
        raise typer.BadParameter("--limit and --task-file cannot be used together.")
    if retry_failed and not resume:
        raise typer.BadParameter("--retry-failed requires --resume.")

    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    selected_task_ids = _load_task_file(task_file) if task_file is not None else None
    task_total = (
        len(selected_task_ids) if selected_task_ids is not None else len(dataset.iter_tasks())
    )
    if limit is not None and selected_task_ids is None:
        task_total = min(task_total, limit)
    if resume and selected_task_ids is None and limit is None:
        task_total = _resume_task_total(app_config) or task_total
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                task_ids=selected_task_ids,
                resume=resume,
                retry_failed=retry_failed,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks selected: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")
    summary = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    console.print(f"Tasks executed: {summary['executed_task_count']}")
    console.print(f"Tasks reused: {summary['resumed_task_count']}")


def main() -> None:
    app()
