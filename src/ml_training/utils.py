"""Shared helpers: paths, seeding, logging and Windows-safe URIs."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(PROJECT_ROOT / "configs")

log = logging.getLogger("ml_training")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def sqlite_uri(uri: str | None) -> str | None:
    """Normalise ``sqlite:///C:\\path\\x.db`` to forward slashes so it works on Windows."""
    if not uri or not uri.startswith("sqlite:///"):
        return uri
    path = Path(uri[len("sqlite:///") :])
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.as_posix()


def file_uri(path: str | Path) -> str:
    """Absolute ``file://`` URI (``file:///D:/...`` on Windows)."""
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p.as_uri()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def flatten_dict(d: dict, parent: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out
