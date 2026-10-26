"""Object-detection datasets in YOLO format (Ultralytics, Roboflow, CVAT, Label Studio exports).

Accepted layouts under ``data/raw/<name>/`` (or ``data.path``):

1. A ``data.yaml`` with ``train`` / ``val`` / ``test`` entries (folders or .txt lists) and ``names``
   – what Roboflow and most tools export (``../train/images`` style paths are handled).
2. ``images/`` + ``labels/`` (flat, or with ``train``/``val``/``test`` sub-folders) and class names
   from ``data.class_names`` or a ``classes.txt``.

``prepare`` puts the raw dataset in ``data/raw/<name>`` (immutable, tracked by DVC).
``featurize`` resolves the splits, creates missing ones (val / test) reproducibly and writes
``data/processed/<name>/{train,val,test}.txt`` + ``data.yaml`` + ``meta.json``.
"""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from ml_training.utils import ensure_dir, log, write_json

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dng", ".mpo"}
SPLIT_ALIASES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}


def configure_ultralytics(cfg: DictConfig) -> None:
    """Keep Ultralytics inside the project and stop its own MLflow/W&B callbacks (we log)."""
    from ultralytics import settings

    settings.update(
        {
            "datasets_dir": str(Path(cfg.paths.raw).resolve()),
            "runs_dir": str((Path(cfg.paths.root) / "models" / "runs").resolve()),
            "weights_dir": str((Path(cfg.paths.models) / "pretrained").resolve()),
            "mlflow": False,
            "wandb": False,
            "tensorboard": False,
            "sync": False,
        }
    )
    # data.utils reads the setting at import time, so point it at data/raw explicitly too.
    import ultralytics.data.utils as data_utils

    data_utils.DATASETS_DIR = Path(cfg.paths.raw).resolve()


def raw_root(cfg: DictConfig) -> Path:
    return Path(cfg.paths.raw) / cfg.data.name


# --------------------------------------------------------------------------- prepare
def prepare(cfg: DictConfig) -> Path:
    d = cfg.data
    if d.source == "ultralytics":  # built-in dataset (coco8, coco128, VOC, ...): auto-download
        configure_ultralytics(cfg)
        from ultralytics.data.utils import check_det_dataset

        info = check_det_dataset(d.loader, autodownload=True)
        root = Path(info["path"])
        log.info("Ultralytics dataset %s ready at %s (%d classes)", d.loader, root, info["nc"])
        return root

    if d.source != "yolo":
        raise ValueError(f"Detection data.source must be 'yolo' or 'ultralytics', got {d.source!r}")
    src = Path(d.path)
    if not src.is_absolute():
        src = Path(cfg.paths.root) / src
    if not src.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {src}\n"
            f"Copy your YOLO dataset (data.yaml + images/labels) to data/raw/{d.name}/"
        )
    dst = raw_root(cfg)
    if src.resolve() != dst.resolve():
        log.info("Copying %s -> %s (raw data is kept immutable under data/raw)", src, dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    n_img = sum(1 for p in dst.rglob("*") if p.suffix.lower() in IMG_EXTS)
    n_lbl = sum(1 for p in dst.rglob("*.txt") if "labels" in p.parts)
    log.info("Raw detection dataset %s: %d images, %d label files", dst, n_img, n_lbl)
    return dst


# --------------------------------------------------------------------------- featurize
def _list_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if path.suffix == ".txt" and path.is_file():
        base = path.parent
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                out.append(p if p.is_absolute() else (base / p).resolve())
        return out
    if path.suffix.lower() in IMG_EXTS and path.is_file():
        return [path]
    return []


def _resolve_entry(entry: Any, root: Path, yaml_dir: Path) -> list[Path]:
    """Resolve a data.yaml split entry; tolerates Roboflow's '../train/images' paths."""
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    for e in entries:
        e = str(e)
        stripped = e
        while stripped.startswith(("../", "..\\", "./", ".\\")):
            stripped = stripped[3:] if stripped.startswith("..") else stripped[2:]
        candidates = [Path(e), root / e, yaml_dir / e, root / stripped, yaml_dir / stripped]
        for c in candidates:
            found = _list_images(c)
            if found:
                images += found
                break
        else:
            log.warning("Split entry %r not found under %s", e, root)
    return images


def img2label(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _names_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names or [])}


def _find_splits(root: Path, d: DictConfig) -> tuple[dict[str, list[Path]], dict[int, str]]:
    yaml_path = root / d.get("yaml_file", "data.yaml")
    if not yaml_path.exists():
        found = sorted(root.rglob("*.yaml"))
        yaml_path = found[0] if found else yaml_path
    splits: dict[str, list[Path]] = {}
    names: dict[int, str] = {}
    if yaml_path.exists():
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        base = Path(spec["path"]) if spec.get("path") else root
        if not base.is_absolute():
            base = (yaml_path.parent / base).resolve()
        if not base.exists():
            base = root
        names = _names_dict(spec.get("names"))
        for split, aliases in SPLIT_ALIASES.items():
            for a in aliases:
                if spec.get(a):
                    splits[split] = _resolve_entry(spec[a], base, yaml_path.parent)
                    break
    else:
        img_dir = root / "images"
        if not img_dir.exists():
            raise FileNotFoundError(f"No data.yaml and no images/ folder in {root}")
        for split, aliases in SPLIT_ALIASES.items():
            for a in aliases:
                if (img_dir / a).is_dir():
                    splits[split] = _list_images(img_dir / a)
                    break
        if not splits:
            splits["train"] = _list_images(img_dir)
    if d.get("class_names"):
        names = _names_dict(list(d.class_names))
    elif not names and (root / "classes.txt").exists():
        lines = (root / "classes.txt").read_text(encoding="utf-8").splitlines()
        names = _names_dict([n.strip() for n in lines if n.strip()])
    if not names:
        raise ValueError("Class names not found: add `names` to data.yaml or set data.class_names")
    return {k: v for k, v in splits.items() if v}, names


def _count(images: list[Path], nc: int) -> dict[str, Any]:
    inst: Counter[int] = Counter()
    background = missing = 0
    for img in images:
        lbl = img2label(img)
        if not lbl.exists():
            missing += 1
            continue
        rows = [r.split() for r in lbl.read_text(encoding="utf-8").splitlines() if r.strip()]
        if not rows:
            background += 1
        for r in rows:
            inst[int(float(r[0]))] += 1
    bad = [c for c in inst if c >= nc]
    if bad:
        raise ValueError(f"Label class ids {bad} >= number of classes {nc}")
    return {
        "images": len(images),
        "background_images": background,
        "missing_labels": missing,
        "instances": {int(k): int(v) for k, v in sorted(inst.items())},
    }


def featurize(cfg: DictConfig) -> Path:
    d = cfg.data
    if d.source == "ultralytics":
        configure_ultralytics(cfg)
        from ultralytics.data.utils import check_det_dataset

        info = check_det_dataset(d.loader, autodownload=True)
        root = Path(info["path"])
        names = _names_dict(info["names"])
        splits = {k: _resolve_entry(info[k], root, root) for k in SPLIT_ALIASES if info.get(k)}
        splits = {k: v for k, v in splits.items() if v}
    else:
        splits, names = _find_splits(raw_root(cfg), d)
    if "train" not in splits:
        raise ValueError(f"No training images found for dataset {d.name}")

    rng = random.Random(cfg.seed)
    pool = list(splits["train"])
    if "val" not in splits:  # only train given: carve val (and test) out of it
        rng.shuffle(pool)
        n_test = int(len(pool) * d.get("test_size", 0)) if "test" not in splits else 0
        n_val = max(1, int(len(pool) * d.get("val_size", 0.2)))
        if n_test:
            splits["test"], pool = pool[:n_test], pool[n_test:]
        splits["val"], pool = pool[:n_val], pool[n_val:]
        splits["train"] = sorted(pool)
        log.info(
            "Created missing splits (seed=%d): %s", cfg.seed, {k: len(v) for k, v in splits.items()}
        )

    out = ensure_dir(cfg.paths.processed)
    spec: dict[str, Any] = {"path": str(out.resolve()), "names": names}
    for split, imgs in splits.items():
        (out / f"{split}.txt").write_text(
            "\n".join(str(p.resolve()) for p in imgs) + "\n", encoding="utf-8"
        )
        spec[split] = f"{split}.txt"
    (out / "data.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")

    counts = {s: _count(imgs, len(names)) for s, imgs in splits.items()}
    meta = {
        "dataset": d.name,
        "task": "detect",
        "class_names": [names[i] for i in sorted(names)],
        "n_classes": len(names),
        "data_yaml": str((out / "data.yaml").resolve()),
        "n_rows": {s: c["images"] for s, c in counts.items()},
        "counts": counts,
        "data_config": OmegaConf.to_container(d, resolve=True),
    }
    write_json(meta, out / "meta.json")
    for s, c in counts.items():
        log.info(
            "%-5s %5d images  %6d boxes  (%d background, %d missing labels)",
            s,
            c["images"],
            sum(c["instances"].values()),
            c["background_images"],
            c["missing_labels"],
        )
    if any(c["missing_labels"] for c in counts.values()):
        log.warning("Some images have no label file – they are treated as background images")
    return out


def is_detection(cfg: DictConfig) -> bool:
    return cfg.data.get("task", "classify") == "detect"
