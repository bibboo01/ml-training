"""Delete generated data / results so a dataset can be re-run from scratch.

uv run python -m ml_training.clean helmet          # results of one dataset (asks first)
uv run python -m ml_training.clean helmet --raw    # ... and its raw copy in data/raw
uv run python -m ml_training.clean all             # every result, MLflow + Optuna databases
uv run python -m ml_training.clean all --dry-run   # only list what would be deleted

Per dataset: data/processed/<name>, models/<name>, reports/<name>, its MLflow experiment
(runs, artifacts, registered models <name>-<model>) and Optuna studies <name>-<model>.
Never touched: models/pretrained (downloaded weights), configs, code, logs/history.jsonl
(the permanent command log – the clean itself is recorded there) and data/raw unless --raw.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from ml_training.utils import PROJECT_ROOT

KEEP = {".gitkeep", "pretrained"}


def _model_names(root: Path) -> set[str]:
    return {p.stem for p in (root / "configs" / "model").glob("*.yaml")}


def _belongs(name: str, dataset: str, models: set[str]) -> bool:
    """'helmet-yolo11n' belongs to helmet; 'coco8-seg-yolo11n' does not belong to coco8."""
    return name.startswith(f"{dataset}-") and name[len(dataset) + 1 :] in models


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _children(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.glob("*")) if p.name not in KEEP] if folder.is_dir() else []


def file_targets(root: Path, target: str, raw: bool) -> list[Path]:
    if target == "all":
        paths = [
            *_children(root / "data" / "processed"),
            *_children(root / "data" / "interim"),
            *_children(root / "models"),
            *_children(root / "reports"),
            *[
                root / n
                for n in ("mlflow.db", "mlruns", "optuna.db", "outputs", "multirun", "runs")
            ],
        ]
        if raw:
            paths += _children(root / "data" / "raw")
    else:
        paths = [
            root / "data" / "processed" / target,
            root / "models" / target,
            root / "reports" / target,
        ]
        if raw:
            paths += [root / "data" / "raw" / target, root / "data" / "raw" / f"{target}.csv"]
    return [p for p in paths if p.exists()]


def _rm(p: Path) -> None:
    def force(func, path, _exc):  # Windows: clear read-only flags and retry
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p, onerror=force)
    else:
        p.unlink()


def clean_tracking(root: Path, dataset: str, dry_run: bool) -> list[str]:
    """Remove one dataset's MLflow experiment / registered models and its Optuna studies."""
    done: list[str] = []
    models = _model_names(root)
    db = root / "mlflow.db"
    if db.exists():
        from mlflow import MlflowClient

        uri = "sqlite:///" + db.as_posix()
        client = MlflowClient(tracking_uri=uri)
        for rm in client.search_registered_models():
            if _belongs(rm.name, dataset, models):
                done.append(f"MLflow registered model {rm.name}")
                if not dry_run:
                    client.delete_registered_model(rm.name)
        exp = client.get_experiment_by_name(dataset)
        if exp is not None:
            done.append(f"MLflow experiment {dataset} (all runs + artifacts)")
            if not dry_run:
                if exp.lifecycle_stage != "deleted":
                    client.delete_experiment(exp.experiment_id)
                # permanent delete, so the next training can create the experiment again
                subprocess.run(
                    [sys.executable, "-m", "mlflow", "gc", "--backend-store-uri", uri,
                     "--experiment-ids", exp.experiment_id],
                    check=True, capture_output=True, cwd=root,
                    env={**os.environ, "MLFLOW_TRACKING_URI": uri},
                )  # fmt: skip
    odb = root / "optuna.db"
    if odb.exists():
        import optuna

        storage = "sqlite:///" + odb.as_posix()
        for s in optuna.get_all_study_summaries(storage=storage):
            if _belongs(s.study_name, dataset, models):
                done.append(f"Optuna study {s.study_name}")
                if not dry_run:
                    optuna.delete_study(study_name=s.study_name, storage=storage)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="dataset name, or 'all'")
    ap.add_argument("--raw", action="store_true", help="also delete the raw data in data/raw")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be deleted")
    ap.add_argument("--root", default=str(PROJECT_ROOT), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = Path(args.root)
    target = args.target.strip().strip("\"'").rstrip("/\\")
    if not target or any(c in target for c in "/\\:") or target in (".", ".."):
        print(f"Give a dataset name (folder name), not a path: {args.target!r}", file=sys.stderr)
        return 2

    files = file_targets(root, target, args.raw)
    tracking = [] if target == "all" else clean_tracking(root, target, dry_run=True)
    if not files and not tracking:
        print(f"Nothing to delete for '{target}'.")
        return 0
    print(f"Will delete ({'everything' if target == 'all' else target}):")
    for p in files:
        print(f"  {p.relative_to(root)}  ({_human(_size(p))})")
    for t in tracking:
        print(f"  {t}")
    if not args.raw:
        print("Kept: data/raw (add --raw to delete it too), models/pretrained, configs, code")
    if args.dry_run:
        return 0
    if not args.yes:
        try:
            answer = input("Type yes to delete: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("yes", "y"):
            print("Cancelled – nothing deleted.")
            return 1
    if tracking:
        clean_tracking(root, target, dry_run=False)
    failed = []
    for p in files:
        try:
            _rm(p)
        except OSError as e:  # e.g. mlflow.db open in the MLflow UI
            failed.append(f"{p.relative_to(root)}: {e}")
    print(f"Deleted {len(files) + len(tracking) - len(failed)} item(s).")
    from ml_training.lineage import record

    record(
        "clean",
        root / "logs" / "history.jsonl",
        target=target,
        raw=args.raw,
        deleted=[str(p.relative_to(root)) for p in files] + tracking,
        failed=failed,
    )
    if failed:
        print("Could not delete (close MLflow UI / Excel and run again):", *failed, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
