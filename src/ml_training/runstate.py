"""models/<dataset>/<model>/run_state.json – where a training is, so `resume` can continue it.

Shared by every backend (YOLO, text, ...). status: running | finished,
stage: start | hpo | final | evaluate | done, overrides: the exact Hydra command settings.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ml_training.utils import read_json, write_json

STATE_FILE = "run_state.json"


def hydra_overrides() -> list[str]:
    try:
        from hydra.core.hydra_config import HydraConfig

        task = HydraConfig.get().overrides.task
    except Exception:  # noqa: BLE001 - called outside a Hydra app (tests)
        return []
    return [o for o in task if not o.lstrip("+").startswith("resume=")]


def read_state(runs_dir: Path) -> dict[str, Any]:
    path = Path(runs_dir) / STATE_FILE
    return read_json(path) if path.exists() else {}


def write_state(runs_dir: Path, **fields: Any) -> dict[str, Any]:
    path = Path(runs_dir) / STATE_FILE
    state = read_state(runs_dir)
    state.update(fields, updated=datetime.now().isoformat(timespec="seconds"))
    write_json(state, path)
    return state


def start(runs_dir: Path, resume: bool, **info: Any) -> dict[str, Any]:
    """Call at the start of a training. Returns the previous state (empty when not resuming)."""
    prev = read_state(runs_dir) if resume else {}
    if resume:
        write_state(runs_dir, status="running", resumes=int(prev.get("resumes", 0)) + 1)
    else:
        write_state(
            runs_dir,
            status="running",
            stage="start",
            overrides=hydra_overrides(),
            started=datetime.now().isoformat(timespec="seconds"),
            resumes=0,
            **info,
        )
    return prev
