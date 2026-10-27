"""Image datasets for every Ultralytics YOLO task (Roboflow, CVAT, Label Studio exports).

Tasks: detect, segment, pose, obb (YOLO .txt labels) and classify (one folder per class).
``data.task: auto`` works out the task from the files:
classify = class sub-folders and no labels, pose = ``kpt_shape`` in data.yaml,
obb = every label row has 9 numbers, segment = polygon rows, detect = 5 numbers per row.

Accepted layouts under ``data/raw/<name>/`` (or ``data.path``) for detect/segment/pose/obb:

1. A ``data.yaml`` with ``train`` / ``val`` / ``test`` entries (folders or .txt lists) and ``names``
   – what Roboflow and most tools export (``../train/images`` style paths are handled).
2. ``images/`` + ``labels/`` (flat, or with ``train``/``val``/``test`` sub-folders) and class names
   from ``data.class_names`` or a ``classes.txt``.

Classify: ``train/<class>/*.jpg`` + ``val|valid|test/<class>/*.jpg``, or just ``<class>/*.jpg``.

``prepare`` puts the raw dataset in ``data/raw/<name>`` (immutable, tracked by DVC).
``featurize`` resolves the splits, creates missing ones (val / test) reproducibly and writes
``data/processed/<name>/{train,val,test}.txt`` + ``data.yaml`` + ``meta.json``
(classify: ``data/processed/<name>/{train,val,test}/<class>/`` hard links to the raw images).
"""

from __future__ import annotations

import os
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
YOLO_TASKS = ("detect", "segment", "pose", "obb", "classify")
# data.yaml keys besides names/splits that Ultralytics needs (pose keypoints, channels ...)
EXTRA_KEYS = ("kpt_shape", "flip_idx", "kpt_names", "channels")


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
    if d.source == "ultralytics":  # built-in dataset (coco8, coco8-seg, dota8, mnist160 ...)
        info = _builtin_info(cfg)
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
            f"Copy your YOLO dataset (data.yaml + images/labels, or class folders) "
            f"to data/raw/{d.name}/"
        )
    dst = raw_root(cfg)
    if src.resolve() != dst.resolve():
        log.info("Copying %s -> %s (raw data is kept immutable under data/raw)", src, dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    n_img = sum(1 for p in dst.rglob("*") if p.suffix.lower() in IMG_EXTS)
    n_lbl = sum(1 for p in dst.rglob("*.txt") if "labels" in p.parts)
    log.info("Raw image dataset %s: %d images, %d label files", dst, n_img, n_lbl)
    return dst


def _builtin_info(cfg: DictConfig) -> dict[str, Any]:
    configure_ultralytics(cfg)
    from ultralytics.data.utils import check_cls_dataset, check_det_dataset

    if cfg.data.get("task") == "classify":
        info = check_cls_dataset(cfg.data.loader)
        info["path"] = str(Path(info["train"]).parent)
        return info
    return check_det_dataset(cfg.data.loader, autodownload=True)


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


def _find_yaml(root: Path, d: DictConfig) -> dict[str, Any] | None:
    yaml_path = root / d.get("yaml_file", "data.yaml")
    if not yaml_path.exists():
        found = sorted(root.rglob("*.yaml"))
        yaml_path = found[0] if found else yaml_path
    if not yaml_path.exists():
        return None
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}


def _find_splits(
    root: Path, d: DictConfig
) -> tuple[dict[str, list[Path]], dict[int, str], dict[str, Any]]:
    yaml_path = root / d.get("yaml_file", "data.yaml")
    if not yaml_path.exists():
        found = sorted(root.rglob("*.yaml"))
        yaml_path = found[0] if found else yaml_path
    splits: dict[str, list[Path]] = {}
    names: dict[int, str] = {}
    extra: dict[str, Any] = {}
    if yaml_path.exists():
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        extra = {k: spec[k] for k in EXTRA_KEYS if spec.get(k) is not None}
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
    return {k: v for k, v in splits.items() if v}, names, extra


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


# --------------------------------------------------------------------------- task detection
def _class_dirs(path: Path) -> list[Path]:
    """Sub-folders of ``path`` that directly contain images (classify layout)."""
    if not path.is_dir():
        return []
    return sorted(c for c in path.iterdir() if c.is_dir() and any(_is_img(f) for f in c.iterdir()))


def _is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _classify_split_dirs(root: Path) -> dict[str, Path]:
    """{split: folder of class sub-folders}; a bare ``<class>/*.jpg`` folder counts as train."""
    found: dict[str, Path] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for a in aliases:
            if _class_dirs(root / a):
                found[split] = root / a
                break
    if not found and _class_dirs(root) and not (root / "images").is_dir():
        found["train"] = root
    return found


def detect_task(root: Path, d: DictConfig) -> str:
    """Work out the YOLO task of a dataset folder (used when data.task is auto)."""
    spec = _find_yaml(root, d)
    has_labels = any(p.suffix == ".txt" and "labels" in p.parts for p in root.rglob("*.txt"))
    if not has_labels and _classify_split_dirs(root):
        return "classify"
    if spec and spec.get("kpt_shape"):
        return "pose"
    widths: set[int] = set()
    for n, lbl in enumerate(p for p in root.rglob("*.txt") if "labels" in p.parts):
        if n >= 500:
            break
        for row in lbl.read_text(encoding="utf-8").splitlines():
            if row.strip():
                widths.add(len(row.split()))
    if not widths or widths == {5}:
        return "detect"
    if widths == {9}:
        return "obb"  # cls + 4 corner points (a 4-point polygon dataset needs data.task=segment)
    return "segment"


def resolve_task(cfg: DictConfig) -> str:
    task = str(cfg.data.get("task") or "auto")
    if task == "auto":
        task = detect_task(raw_root(cfg), cfg.data)
        log.info("data.task=auto -> detected task: %s  (override with data.task=...)", task)
    if task not in YOLO_TASKS:
        raise ValueError(f"data.task must be auto or one of {YOLO_TASKS}, got {task!r}")
    return task


# --------------------------------------------------------------------------- featurize
def _carve_splits(splits: dict[str, list], d: DictConfig, rng: random.Random) -> dict[str, list]:
    """Only train given: carve val (and test) out of it, reproducibly."""
    if "val" in splits:
        return splits
    pool = list(splits["train"])
    rng.shuffle(pool)
    n_test = int(len(pool) * d.get("test_size", 0)) if "test" not in splits else 0
    n_val = max(1, int(len(pool) * d.get("val_size", 0.2)))
    if n_test:
        splits["test"], pool = pool[:n_test], pool[n_test:]
    splits["val"], pool = pool[:n_val], pool[n_val:]
    splits["train"] = sorted(pool)
    return splits


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # no extra disk space (same drive)
    except OSError:
        shutil.copy2(src, dst)


def _featurize_classify(cfg: DictConfig, root: Path) -> tuple[Path, dict[str, Any]]:
    d = cfg.data
    dirs = _classify_split_dirs(root)
    if "train" not in dirs:
        raise FileNotFoundError(
            f"Classification data needs class folders in {root}: train/<class>/*.jpg "
            "(+ val/<class>/...) or simply <class>/*.jpg"
        )
    class_names = [c.name for c in _class_dirs(dirs["train"])]
    names = dict(enumerate(class_names))
    by_split: dict[str, list[tuple[Path, int]]] = {}
    for split, folder in dirs.items():
        items = []
        for c in _class_dirs(folder):
            if c.name not in class_names:
                log.warning("Class folder %s/%s is not in train – skipped", split, c.name)
                continue
            items += [(f, class_names.index(c.name)) for f in sorted(c.iterdir()) if _is_img(f)]
        by_split[split] = items
    if "val" not in by_split:  # stratified: carve val/test per class
        rng = random.Random(cfg.seed)
        parts: dict[str, list] = {}
        for k in range(len(class_names)):
            per = {"train": [x for x in by_split["train"] if x[1] == k]}
            if "test" in by_split:
                per["test"] = []
            for split, v in _carve_splits(per, d, rng).items():
                parts.setdefault(split, []).extend(v)
        if "test" in by_split:
            parts["test"] = by_split["test"]
        by_split = {k: v for k, v in parts.items() if v}
        log.info(
            "Created missing splits (seed=%d): %s",
            cfg.seed,
            {k: len(v) for k, v in by_split.items()},
        )

    out = ensure_dir(cfg.paths.processed)
    for split in ("train", "val", "test"):
        shutil.rmtree(out / split, ignore_errors=True)
    for split, items in by_split.items():
        for c in class_names:
            ensure_dir(out / split / c)
        for f, k in items:
            _link(f, out / split / class_names[k] / f.name)
    counts = {
        s: {
            "images": len(items),
            "background_images": 0,
            "missing_labels": 0,
            "instances": {k: n for k, n in sorted(Counter(k for _, k in items).items())},
        }
        for s, items in by_split.items()
    }
    return out, {"names": names, "counts": counts, "data": str(out.resolve()), "extra": {}}


def featurize(cfg: DictConfig) -> Path:
    d = cfg.data
    if d.source == "ultralytics":
        info = _builtin_info(cfg)
        root = Path(info["path"])
        task = str(d.get("task") or "detect")
    else:
        root = raw_root(cfg)
        task = resolve_task(cfg)

    if task == "classify":
        out, res = _featurize_classify(cfg, root)
        names, counts, data_ref = res["names"], res["counts"], res["data"]
    else:
        if d.source == "ultralytics":
            names = _names_dict(info["names"])
            extra = {k: info[k] for k in EXTRA_KEYS if info.get(k) is not None}
            splits = {k: _resolve_entry(info[k], root, root) for k in SPLIT_ALIASES if info.get(k)}
            splits = {k: v for k, v in splits.items() if v}
        else:
            splits, names, extra = _find_splits(root, d)
        if "train" not in splits:
            raise ValueError(f"No training images found for dataset {d.name}")
        had_val = "val" in splits
        splits = _carve_splits(splits, d, random.Random(cfg.seed))
        if not had_val:
            log.info(
                "Created missing splits (seed=%d): %s",
                cfg.seed,
                {k: len(v) for k, v in splits.items()},
            )
        out = ensure_dir(cfg.paths.processed)
        spec: dict[str, Any] = {"path": str(out.resolve()), "names": names, **extra}
        for split, imgs in splits.items():
            (out / f"{split}.txt").write_text(
                "\n".join(str(p.resolve()) for p in imgs) + "\n", encoding="utf-8"
            )
            spec[split] = f"{split}.txt"
        (out / "data.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
        data_ref = str((out / "data.yaml").resolve())
        counts = {s: _count(imgs, len(names)) for s, imgs in splits.items()}

    meta = {
        "dataset": d.name,
        "task": task,
        "class_names": [names[i] for i in sorted(names)],
        "n_classes": len(names),
        "data_yaml": data_ref,  # data.yaml (detect/segment/pose/obb) or folder (classify)
        "n_rows": {s: c["images"] for s, c in counts.items()},
        "counts": counts,
        "data_config": OmegaConf.to_container(d, resolve=True),
    }
    write_json(meta, out / "meta.json")
    unit = "images" if task == "classify" else "objects"
    log.info("Task: %s  classes: %d", task, len(names))
    for s, c in counts.items():
        log.info(
            "%-5s %5d images  %6d %s  (%d background, %d missing labels)",
            s,
            c["images"],
            sum(c["instances"].values()),
            unit,
            c["background_images"],
            c["missing_labels"],
        )
    if any(c["missing_labels"] for c in counts.values()):
        log.warning("Some images have no label file – they are treated as background images")
    return out


def is_yolo(cfg: DictConfig) -> bool:
    """Image dataset for Ultralytics YOLO (any task) – as opposed to tabular data."""
    return cfg.data.get("source") in ("yolo", "ultralytics")


is_detection = is_yolo  # backwards-compatible name
