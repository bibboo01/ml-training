"""Lineage / history: everything needed to know later how a result was produced.

Per MLflow run (``log_context``): git commit + uncommitted diff, command line, user / host,
Python + library versions, CPU / RAM / GPU, uv.lock + pyproject.toml, and a fingerprint
(sha256 over file contents) of the exact data used – plus the split file lists.

Across runs (``record``): one JSON line per command in ``logs/history.jsonl``
(prepare, featurize, train, predict, xai, clean) – never deleted by ``clean``.
"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ml_training.utils import PROJECT_ROOT, log

PACKAGES = [
    "ml-training", "torch", "torchvision", "ultralytics", "lightning", "scikit-learn",
    "mlflow", "optuna", "hydra-core", "numpy", "pandas", "dvc", "xlsxwriter",
]  # fmt: skip
HISTORY = PROJECT_ROOT / "logs" / "history.jsonl"


# --------------------------------------------------------------------------- facts
def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], root)
    if not commit:
        return {"available": False}
    status = _run(["git", "status", "--porcelain"], root) or ""
    return {
        "available": True,
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(status),
        "changed_files": [line[3:] for line in status.splitlines()][:200],
    }


def git_diff(root: Path = PROJECT_ROOT) -> str:
    return _run(["git", "diff", "HEAD", "--", "src", "configs", "pipelines"], root) or ""


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "user": _user(),
        "packages": {},
    }
    for pkg in PACKAGES:
        with contextlib.suppress(metadata.PackageNotFoundError):
            info["packages"][pkg] = metadata.version(pkg)
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpus"] = [
                f"{torch.cuda.get_device_name(i)} "
                f"({torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GB)"
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name in some containers
        return "unknown"


def fingerprint(files: Iterable[Path], root: Path | None = None) -> dict[str, Any]:
    """sha256 over (relative name, content) of every file – same data => same hash."""
    h = hashlib.sha256()
    n = size = 0
    for f in sorted({Path(p) for p in files}, key=lambda p: p.as_posix()):
        if not f.is_file():
            continue
        try:
            name = f.relative_to(root).as_posix() if root else f.name
        except ValueError:
            name = f.name
        h.update(name.encode())
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        n += 1
        size += f.stat().st_size
    return {"sha256": h.hexdigest(), "files": n, "bytes": size}


def yolo_data_files(processed: Path, task: str) -> list[Path]:
    """Images + label files the YOLO run actually used (from the processed split lists)."""
    if task == "classify":
        return [p for p in processed.rglob("*") if p.is_file() and p.suffix != ".cache"]
    from ml_training.detection_data import img2label

    files: list[Path] = []
    for lst in processed.glob("*.txt"):
        for line in lst.read_text(encoding="utf-8").splitlines():
            if line.strip():
                img = Path(line.strip())
                files += [img, img2label(img)]
    return files


# --------------------------------------------------------------------------- mlflow
def log_context(
    data_files: Iterable[Path] = (),
    data_root: Path | None = None,
    extra_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    """Log lineage into the active MLflow run; returns a short summary for reports."""
    import mlflow

    git = git_info()
    env = env_info()
    data = fingerprint(data_files, data_root) if data_files else {}
    tags = {
        "command": " ".join(sys.argv)[:5000],
        "host": env["hostname"],
        "user": env["user"],
        "python": env["python"],
        "gpu": ", ".join(env.get("gpus", [])) or "none",
    }
    if git.get("available"):
        tags |= {"git.commit": git["commit"], "git.branch": str(git["branch"]),
                 "git.dirty": str(git["dirty"])}  # fmt: skip
    if data:
        tags |= {"data.sha256": data["sha256"], "data.files": str(data["files"])}
    mlflow.set_tags(tags)
    mlflow.log_dict(env, "lineage/environment.json")
    mlflow.log_dict(git, "lineage/git.json")
    if data:
        mlflow.log_dict(data, "lineage/data_fingerprint.json")
    if git.get("dirty"):
        diff = git_diff()
        if diff:
            mlflow.log_text(diff, "lineage/uncommitted.diff")
    for f in [PROJECT_ROOT / "uv.lock", PROJECT_ROOT / "pyproject.toml", *extra_artifacts]:
        if Path(f).is_file():
            mlflow.log_artifact(str(f), artifact_path="lineage")
    return {
        "git commit": (git.get("commit", "")[:10] + (" (+uncommitted)" if git.get("dirty") else ""))
        if git.get("available")
        else "no git",
        "data sha256": data.get("sha256", "")[:16] or "-",
        "machine": f"{env['hostname']} · {tags['gpu'] if tags['gpu'] != 'none' else 'CPU'}",
    }


def log_epochs(results_csv: Path) -> int:
    """Ultralytics results.csv -> MLflow step metrics (one curve per column)."""
    import mlflow
    import pandas as pd

    if not results_csv.exists():
        return 0
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    cols = [c for c in df.columns if c not in ("epoch", "time")]
    safe = {
        c: "epoch/" + "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in c) for c in cols
    }
    for i, row in df.iterrows():
        step = int(row["epoch"]) if "epoch" in df else int(i) + 1
        mlflow.log_metrics(
            {safe[c]: float(row[c]) for c in cols if pd.notna(row[c])}, step=step, synchronous=False
        )
    mlflow.flush_async_logging()
    return len(df)


# --------------------------------------------------------------------------- history file
def history_path(cfg) -> Path | None:
    h = cfg.get("mlflow", {}).get("history_file") if cfg is not None else None
    return Path(h) if h else None


def record(event: str, history: Path | None = None, **fields: Any) -> None:
    """Append one line to logs/history.jsonl – a permanent log of every command."""
    path = history or HISTORY
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "user": _user(),
        "host": socket.gethostname(),
        "command": " ".join(sys.argv),
        **fields,
    }
    commit = git_info().get("commit")
    if commit:
        entry["git_commit"] = commit[:10]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as e:  # history must never break a run
        log.warning("Could not write %s: %s", path, e)


# --------------------------------------------------------------------------- viewer
def main(argv: list[str] | None = None) -> int:
    """uv run python -m ml_training.lineage [N] [--excel] – show the last N history lines."""
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description="Show logs/history.jsonl")
    ap.add_argument("n", nargs="?", type=int, default=20)
    ap.add_argument("--excel", action="store_true", help="also write logs/history.xlsx")
    ap.add_argument("--file", default=str(HISTORY), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"No history yet ({path})")
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    df = pd.json_normalize(rows)
    show = [c for c in ("time", "event", "dataset", "model", "task", "tune") if c in df]
    show += [c for c in ("metrics.map50_95", "metrics.top1", "metrics.f1") if c in df]
    show += [c for c in ("n_predictions", "target") if c in df]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[show].tail(args.n).astype(object).fillna("").to_string(index=False))
    print(f"\n{len(df)} entries in {path}")
    if args.excel:
        xlsx = path.with_suffix(".xlsx")
        df.to_excel(xlsx, index=False)
        print(f"Excel: {xlsx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
