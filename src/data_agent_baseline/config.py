from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
