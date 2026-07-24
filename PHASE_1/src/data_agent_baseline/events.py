from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EventSink = Callable[[str, dict[str, Any]], None]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JsonlEventRecorder:
    def __init__(self, path: Path, *, task_id: str) -> None:
        self.path = path
        self.task_id = task_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "timestamp": utc_now_iso(),
            "task_id": self.task_id,
            "event_type": event_type,
            **(payload or {}),
        }
        rendered = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())


def emit_event(
    event_sink: EventSink | None,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if event_sink is not None:
        event_sink(event_type, payload or {})


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events
