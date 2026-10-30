"""Tuning presets (configs/tune/*.yaml): list, create from questions, check.

uv run python -m ml_training.tune_menu list            # table of every preset
uv run python -m ml_training.tune_menu create          # ask the common settings, save a preset
uv run python -m ml_training.tune_menu check odb_gpu   # validate a preset you edited by hand

Only the settings people change most are asked; everything else is copied from the base preset
and can still be edited in the saved file.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from ml_training.i18n import RUN
from ml_training.utils import CONFIG_DIR

TUNE_DIR = Path(CONFIG_DIR) / "tune"
BUILTIN = {
    "default": "model config as it is",
    "fast": "50 epochs, small search",
    "balanced": "optimiser + loss + augmentation search",
    "accurate": "300 epochs, widest search (GPU)",
    "small_data": "frozen backbone + strong augmentation",
    "freeze": "transfer learning, tune learning rate only",
    "augment": "search augmentation only",
    "custom": "every setting, edit the file",
}
OPTIMIZERS = ["auto", "SGD", "AdamW", "Adam", "NAdam", "RAdam", "RMSProp"]
# (key in the composed config, question, kind)
QUESTIONS = [
    ("model.params.epochs", "Epochs for the final model", "int"),
    ("model.params.imgsz", "Image size (640 normal, 960+ for small objects)", "int"),
    ("model.params.batch", "Batch size (-1 = auto for your GPU)", "int"),
    ("model.params.patience", "Stop early after N epochs without improvement", "int"),
    ("model.params.freeze", "Freeze first N layers (0 = train all, 10 = backbone)", "int"),
    ("model.params.optimizer", f"Optimizer ({' / '.join(OPTIMIZERS)})", "optimizer"),
    ("model.params.device", "Device (auto / cpu / 0 / 0,1)", "device"),
    ("hpo.n_trials", "Optuna trials (0 = no tuning)", "int"),
    ("model.hpo_epochs", "Epochs per Optuna trial", "int"),
]


def presets() -> list[str]:
    names = sorted(p.stem for p in TUNE_DIR.glob("*.yaml"))
    order = [n for n in BUILTIN if n in names]
    return order + [n for n in names if n not in BUILTIN]


def compose(preset: str, model: str = "yolo11n"):
    from hydra import compose as hcompose
    from hydra import initialize_config_dir

    with initialize_config_dir(CONFIG_DIR, version_base="1.3"):
        return hcompose("config", overrides=["data=coco8", f"model={model}", f"tune={preset}"])


def _get(cfg, dotted: str) -> Any:
    node = cfg
    for k in dotted.split("."):
        node = node.get(k) if node is not None else None
    return node


def check(preset: str) -> tuple[bool, str]:
    """Compose the preset and sample its search space once."""
    import optuna

    from ml_training.hpo import suggest_params

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        cfg = compose(preset)
        space = cfg.tune.search_space or cfg.model.search_space
        suggest_params(optuna.create_study().ask(), space)
    except Exception as e:  # noqa: BLE001 - report any config problem to the user
        return False, f"{type(e).__name__}: {e}"
    return True, f"OK – {len(space or {})} settings tuned, {cfg.hpo.n_trials} trials"


def table() -> str:
    rows = [("preset", "epochs", "trials", "freeze", "tuned", "about")]
    for name in presets():
        try:
            cfg = compose(name)
            space = cfg.tune.search_space or cfg.model.search_space or {}
            freeze = cfg.model.params.get("freeze") or "-"
            rows.append(
                (name, str(cfg.model.params.epochs), str(cfg.hpo.n_trials), str(freeze),
                 str(len(space)), BUILTIN.get(name, "your preset"))
            )  # fmt: skip
        except Exception as e:  # noqa: BLE001
            rows.append((name, "?", "?", "?", "?", f"ERROR {type(e).__name__}"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows)


def _parse(kind: str, text: str) -> Any:
    if kind == "int":
        return int(text)
    if kind == "optimizer":
        match = [o for o in OPTIMIZERS if o.lower() == text.lower()]
        if not match:
            raise ValueError(f"choose one of {OPTIMIZERS}")
        return match[0]
    if kind == "device":
        t = text.lower().replace(" ", "")
        if t in ("auto", "none", "null"):
            return None
        if t == "cpu" or re.fullmatch(r"\d+(,\d+)*", t):
            return t if ("," in t or t == "cpu") else int(t)
        raise ValueError("use auto, cpu, 0 or 0,1")
    raise ValueError(kind)


def _ask(question: str, current: Any, kind: str, read=input) -> Any:
    shown = "auto" if current is None and kind == "device" else current
    if kind == "int" and current is None:
        shown = 0
    while True:
        text = read(f"  {question} [{shown}]: ").strip()
        if not text:
            return current if not (kind == "int" and current is None) else 0
        try:
            return _parse(kind, text)
        except ValueError as e:
            print(f"    ! {e}")


def _set(tree: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        tree = tree.setdefault(k, {})
    tree[keys[-1]] = value


def build_preset(base: str, answers: dict[str, Any], name: str) -> dict:
    """Base preset file + answers -> new preset content (search space copied explicitly)."""
    raw = yaml.safe_load((TUNE_DIR / f"{base}.yaml").read_text(encoding="utf-8")) or {}
    out = copy.deepcopy(raw)
    out.setdefault("tune", {})["name"] = name
    if not out["tune"].get("search_space"):
        cfg = compose(base)
        from omegaconf import OmegaConf

        out["tune"]["search_space"] = OmegaConf.to_container(cfg.model.search_space)
    for key, value in answers.items():
        if key == "model.params.freeze" and value == 0:
            value = None
        _set(out, key, value)
    return out


class _Flow(dict):
    """Dumped on one line: one search-space entry per line keeps the file readable."""


yaml.SafeDumper.add_representer(
    _Flow, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v, flow_style=True)
)


def save(content: dict, name: str, base: str) -> Path:
    path = TUNE_DIR / f"{name}.yaml"
    header = (
        "# @package _global_\n"
        f"# Your tuning preset '{name}' (made from '{base}' with the tuning menu).\n"
        f"# Use it with  tune={name}  or pick it in the run.bat menu. Edit freely:\n"
        "# every setting of https://docs.ultralytics.com/modes/train/#train-settings works under\n"
        "# model.params, and search_space decides what Optuna tries.\n"
    )
    content = copy.deepcopy(content)
    space = content.get("tune", {}).get("search_space") or {}
    content["tune"]["search_space"] = {k: _Flow(v) for k, v in space.items()}
    body = yaml.safe_dump(content, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(header + body, encoding="utf-8")
    return path


def create(read=input) -> int:
    names = presets()
    print("Start from which preset?")
    for i, n in enumerate(names, 1):
        print(f"  {i}) {n:<11} {BUILTIN.get(n, 'your preset')}")
    while True:
        t = read("Choose [1]: ").strip() or "1"
        if t.isdigit() and 1 <= int(t) <= len(names):
            base = names[int(t) - 1]
            break
        print("    ! type a number from the list")
    cfg = compose(base)
    print(f"\nPress Enter to keep the value from '{base}':")
    answers = {key: _ask(q, _get(cfg, key), kind, read) for key, q, kind in QUESTIONS}
    while True:
        name = read("\nName for your preset (letters, numbers, _ ): ").strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            print("    ! use letters, numbers and _ only")
            continue
        if name in BUILTIN:
            print("    ! that is a built-in preset – pick another name")
            continue
        if (TUNE_DIR / f"{name}.yaml").exists() and read(
            f"'{name}' exists – overwrite? (y/n) [n]: "
        ).strip().lower() != "y":
            continue
        break
    path = save(build_preset(base, answers, name), name, base)
    ok, msg = check(name)
    print(f"\nSaved {path}\n{'Checked' if ok else 'PROBLEM'}: {msg}")
    print(f"Train with it:  {RUN} fit <dataset> <size> full {name}   (or pick it in the menu)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("create")
    sub.add_parser("names")
    c = sub.add_parser("check")
    c.add_argument("name")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        print(table())
    elif args.cmd == "names":
        for n in presets():
            print(f"{n}|{BUILTIN.get(n, 'your preset')}")
    elif args.cmd == "create":
        return create()
    elif args.cmd == "check":
        ok, msg = check(args.name)
        print(msg)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
