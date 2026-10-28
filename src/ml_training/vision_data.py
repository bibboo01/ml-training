"""Image datasets for the vision backend (timm classifiers, SegFormer semantic segmentation).

Tasks (``data.task: auto`` works it out from the folders):
  classify  one label per image           <class>/*.jpg, or train|val|test/<class>/*.jpg
  segment   a class for every pixel       images/ + masks/ (same file names, mask = PNG where the
            (semantic segmentation)       pixel value is the class index 0..N-1; 0 = background)

Segment layouts: ``images/`` + ``masks/`` (flat, or with train / val / test sub-folders), or
Roboflow "Semantic Segmentation Masks" export (``train/x.jpg`` + ``train/x_mask.png`` and
``_classes.csv``). Class names: data.class_names, ``classes.txt`` or ``_classes.csv``.
Images are not copied again: ``featurize`` writes ``data/processed/<name>/{train,val,test}.jsonl``
(paths + labels) and ``meta.json``.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ml_training.text_data import split_records
from ml_training.utils import ensure_dir, log, read_json, write_json

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
MASK_DIRS = ("masks", "mask", "labels", "annotations", "SegmentationClass")
TASKS = ("classify", "segment")


def raw_root(cfg: DictConfig) -> Path:
    return Path(cfg.paths.raw) / cfg.data.name


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())


def _mask_dir(root: Path) -> Path | None:
    return next((root / d for d in MASK_DIRS if (root / d).is_dir()), None)


def detect_task(root: Path) -> str:
    if _mask_dir(root) or any(p.stem.endswith("_mask") for p in root.rglob("*.png")):
        return "segment"
    return "classify"


# --------------------------------------------------------------------------- classify
def classify_records(root: Path) -> list[dict[str, Any]]:
    recs = []
    for img in _images(root):
        parts = img.relative_to(root).parts[:-1]
        if not parts:
            continue
        split = SPLITS.get(parts[0].lower()) if len(parts) > 1 else None
        label = parts[1] if split else parts[0]
        rec = {"image": str(img.resolve()), "label": label}
        if split:
            rec["split"] = split
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- segment
def segment_records(root: Path) -> list[dict[str, Any]]:
    recs = []
    mdir = _mask_dir(root)
    if mdir is not None:  # images/ + masks/
        idir = root / "images" if (root / "images").is_dir() else root
        masks = {m.relative_to(mdir).with_suffix("").as_posix(): m for m in _images(mdir)}
        for img in _images(idir):
            if mdir in img.parents:
                continue
            key = img.relative_to(idir).with_suffix("").as_posix()
            m = masks.get(key) or masks.get(key + "_mask")
            if m is None:
                continue
            first = img.relative_to(idir).parts[0].lower()
            rec = {"image": str(img.resolve()), "mask": str(m.resolve())}
            if first in SPLITS and len(img.relative_to(idir).parts) > 1:
                rec["split"] = SPLITS[first]
            recs.append(rec)
    else:  # Roboflow: x.jpg + x_mask.png side by side
        for m in sorted(root.rglob("*_mask.png")):
            img = next(
                (
                    m.with_name(m.name[: -len("_mask.png")] + e)
                    for e in IMG_EXTS
                    if m.with_name(m.name[: -len("_mask.png")] + e).exists()
                ),
                None,
            )
            if img is None:
                continue
            rec = {"image": str(img.resolve()), "mask": str(m.resolve())}
            split = SPLITS.get(m.relative_to(root).parts[0].lower())
            if split and len(m.relative_to(root).parts) > 1:
                rec["split"] = split
            recs.append(rec)
    return recs


def segment_classes(root: Path, recs: list[dict], given) -> list[str]:
    if given:
        return [str(c) for c in given]
    for name in ("_classes.csv", "classes.csv"):
        f = next(root.rglob(name), None)
        if f:
            rows = list(csv.reader(f.read_text(encoding="utf-8-sig").splitlines()))
            body = rows[1:] if rows and not rows[0][0].strip().isdigit() else rows
            return [r[-1].strip() for r in sorted(body, key=lambda r: int(r[0]))]
    f = next(root.rglob("classes.txt"), None)
    if f:
        return [x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    import numpy as np
    from PIL import Image

    top = 0
    for r in recs[:200]:
        top = max(top, int(np.asarray(Image.open(r["mask"])).max()))
    return ["background"] + [f"class_{i}" for i in range(1, top + 1)]


# --------------------------------------------------------------------------- pipeline
def _source(cfg: DictConfig) -> Path:
    sample = cfg.data.get("sample")
    if sample and not raw_root(cfg).exists():
        from ml_training.vision_samples import make

        return make(sample, raw_root(cfg))
    p = cfg.data.get("path")
    src = Path(p) if p else raw_root(cfg)
    if not src.is_absolute():
        src = Path(cfg.paths.root) / src
    if not src.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {src} – put images in data/raw/{cfg.data.name}/"
        )
    return src


def prepare(cfg: DictConfig) -> Path:
    src, dst = _source(cfg), raw_root(cfg)
    if src.resolve() != dst.resolve():
        log.info("Copying %s -> %s (raw data is kept immutable under data/raw)", src, dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    log.info("Raw image dataset %s: %d images", dst, len(_images(dst)))
    return dst


def featurize(cfg: DictConfig) -> Path:
    d = cfg.data
    root = raw_root(cfg) if raw_root(cfg).exists() else _source(cfg)
    task = str(d.get("task") or "auto")
    if task == "auto":
        task = detect_task(root)
    if task not in TASKS:
        raise ValueError(f"data.task must be auto, classify or segment – got {task!r}")
    recs = segment_records(root) if task == "segment" else classify_records(root)
    if not recs:
        raise ValueError(f"No {task} images found in {root} (see vision_data.py for layouts)")
    splits = split_records(
        recs,
        float(d.get("val_size", 0.15)),
        float(d.get("test_size", 0.15)),
        int(cfg.seed),
        task == "classify",
    )
    out = ensure_dir(cfg.paths.processed)
    for old in out.glob("*.jsonl"):
        old.unlink()
    for split, rows in splits.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps({k: v for k, v in r.items() if k != "split"}) + "\n" for r in rows),
            encoding="utf-8",
        )
    if task == "segment":
        names = segment_classes(root, recs, d.get("class_names"))
        counts = {s: {"images": len(rows)} for s, rows in splits.items()}
    else:
        names = list(d.get("class_names") or sorted({r["label"] for r in recs}))
        counts = {s: dict(Counter(r["label"] for r in rows)) for s, rows in splits.items()}
    meta = {
        "dataset": d.name,
        "kind": "vision",
        "task": task,
        "class_names": names,
        "n_classes": len(names),
        "n_rows": {s: len(r) for s, r in splits.items()},
        "counts": counts,
        "data_config": OmegaConf.to_container(d, resolve=True),
    }
    write_json(meta, out / "meta.json")
    log.info("Task: %s  classes: %d  images: %s", task, len(names), meta["n_rows"])
    return out


def load_split(processed: Path, split: str) -> list[dict[str, Any]]:
    p = Path(processed) / f"{split}.jsonl"
    return (
        [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
        if p.exists()
        else []
    )


def load_meta(processed: Path) -> dict[str, Any]:
    return read_json(Path(processed) / "meta.json")
