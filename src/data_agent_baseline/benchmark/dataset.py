from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

TASK_DIR_PREFIX = "task_"


def _task_number(task_id: str) -> int:
    if not task_id.startswith(TASK_DIR_PREFIX):
        raise ValueError(f"Invalid task id: {task_id}")
    return int(task_id.removeprefix(TASK_DIR_PREFIX))


def _load_task_record(task_json_path: Path) -> TaskRecord:
    payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    expected_keys = {"task_id", "difficulty", "question"}
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        raise ValueError(
            f"Unexpected task.json keys for {task_json_path.parent.name}: "
            f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )

    return TaskRecord(
        task_id=str(payload["task_id"]),
        difficulty=str(payload["difficulty"]),
        question=str(payload["question"]),
    )


@dataclass(frozen=True, slots=True)
class DABenchPublicDataset:
    root_dir: Path

    @property
    def exists(self) -> bool:
        return self.root_dir.is_dir()

    def task_dirs(self) -> list[Path]:
        if not self.exists:
            return []

        task_dirs = [
            path
            for path in self.root_dir.iterdir()
            if path.is_dir() and path.name.startswith(TASK_DIR_PREFIX)
        ]
        task_dirs.sort(key=lambda path: _task_number(path.name))
        return task_dirs

    def list_task_ids(self) -> list[str]:
        return [path.name for path in self.task_dirs()]

    def get_task(self, task_id: str) -> PublicTask:
        task_dir = self.root_dir / task_id
        task_json_path = task_dir / "task.json"
        if not task_json_path.exists():
            raise FileNotFoundError(f"Missing task.json: {task_json_path}")

        record = _load_task_record(task_json_path)
        if record.task_id != task_dir.name:
            raise ValueError(f"task_id mismatch for {task_dir}: task.json has {record.task_id}")

        context_dir = task_dir / "context"
        if not context_dir.is_dir():
            raise FileNotFoundError(f"Missing context dir: {context_dir}")

        assets = TaskAssets(task_dir=task_dir, context_dir=context_dir)
        return PublicTask(record=record, assets=assets)

    def iter_tasks(
        self,
        *,
        task_ids: list[str] | None = None,
        difficulty: str | None = None,
        difficulties: list[str] | None = None,
    ) -> list[PublicTask]:
        selected_ids = set(task_ids or [])
        selected_difficulties = set(difficulties or [])
        if difficulty is not None:
            selected_difficulties.add(difficulty)

        tasks: list[PublicTask] = []
        for task_dir in self.task_dirs():
            if selected_ids and task_dir.name not in selected_ids:
                continue
            task = self.get_task(task_dir.name)
            if selected_difficulties and task.difficulty not in selected_difficulties:
                continue
            tasks.append(task)
        return tasks

    def task_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.iter_tasks():
            counts[task.difficulty] = counts.get(task.difficulty, 0) + 1
        return counts
