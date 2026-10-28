"""Continue YOLO trainings that stopped (power cut, Ctrl+C, crash, PC restarted).

uv run python -m ml_training.resume list                  # unfinished trainings
uv run python -m ml_training.resume run helmet            # continue (asks if several models)
uv run python -m ml_training.resume run helmet yolo11s

Every YOLO training writes models/<dataset>/<model>/run_state.json with the exact command
settings. `run` repeats them with resume=true: Optuna trials already finished are kept and an
interrupted final training continues from its last saved epoch (last.pt).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ml_training.runstate import STATE_FILE
from ml_training.utils import PROJECT_ROOT, read_json

MODELS = PROJECT_ROOT / "models"


def states(models: Path = MODELS) -> list[dict]:
    out = []
    for f in sorted(models.glob(f"*/*/{STATE_FILE}")):
        s = read_json(f)
        s["dir"] = f.parent
        s.setdefault("dataset", f.parent.parent.name)
        s.setdefault("model", f.parent.name)
        out.append(s)
    return out


def describe(s: dict) -> str:
    stage = s.get("stage", "?")
    if stage == "final":
        final = Path(s["dir"]) / "final"
        cks = sorted(final.glob("checkpoint-*"), key=lambda c: int(c.name.split("-")[-1]))
        if cks:  # HuggingFace (text) checkpoints
            last = cks[-1]
        elif (final / "last.pt").exists():  # vision (timm / SegFormer)
            last = final / "last.pt"
        else:
            from ml_training.modeling.yolo import unfinished_checkpoint

            last = unfinished_checkpoint(final)
        stage = "final training" + (f" (continue from {last.name})" if last else " (restarts)")
    elif stage == "hpo":
        stage = "Optuna search (finished trials are kept)"
    elif stage == "evaluate":
        stage = "evaluation (training is done, only the test is repeated)"
    return f"{s['dataset']}|{s['model']}|{stage}|{s.get('updated', '')}"


def unfinished(models: Path = MODELS) -> list[dict]:
    return [s for s in states(models) if s.get("status") != "finished"]


def command(s: dict) -> list[str]:
    overrides = list(s.get("overrides") or [f"data={s['dataset']}", f"model={s['model']}"])
    return [sys.executable, "-m", "ml_training.modeling.train", *overrides, "resume=true"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("dataset")
    r.add_argument("model", nargs="?")
    args = ap.parse_args(argv)
    todo = unfinished()
    if args.cmd == "list":
        for s in todo:
            print(describe(s))
        return 0
    match = [s for s in todo if s["dataset"] == args.dataset]
    if args.model:
        match = [s for s in match if s["model"] == args.model]
    if not match:
        print(f"Nothing to resume for '{args.dataset}' {args.model or ''}".rstrip())
        return 1
    if len(match) > 1:
        print("Several unfinished models – add the model name:")
        for s in match:
            print("  " + describe(s).replace("|", "  "))
        return 1
    cmd = command(match[0])
    print("Resuming:", " ".join(cmd[1:]))
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    sys.exit(main())
