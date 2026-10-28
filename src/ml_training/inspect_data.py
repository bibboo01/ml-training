"""Look at a folder or file and say what it is, which task fits and which command trains it.

Used by the "train automatically" menu of run.ps1 (drag a folder / file in, no settings).
    python -m ml_training.inspect_data D:/data/helmet          # human summary (Thai)
    python -m ml_training.inspect_data D:/data/sales.xlsx --json

kind   = yolo (labelled boxes / polygons / keypoints) | vision (image classes or masks)
       | text (texts, NER, document folders) | table (numbers / categories in columns)
task   = the task inside that kind (detect, classify, segment, regress, forecast ...)
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from ml_training.text_data import (
    CONLL_EXTS,
    DOC_EXTS,
    IMG_EXTS,
    LABEL_NAMES,
    TABLE_EXTS,
    TEXT_NAMES,
    _input_file,
    _is_conll,
    read_table,
)

TARGET_NAMES = LABEL_NAMES + (
    "price",
    "sales",
    "amount",
    "value",
    "revenue",
    "demand",
    "score",
    "churn",
    "ราคา",
    "ยอดขาย",
    "จำนวน",
)
TASK_TH = {
    "detect": "ตรวจจับวัตถุ (กรอบสี่เหลี่ยม)",
    "segment": "แยกวัตถุเป็นรูปร่าง (polygon)",
    "pose": "จุดสำคัญบนร่างกาย / วัตถุ (pose)",
    "obb": "ตรวจจับแบบกรอบเอียง (obb)",
    "classify": "จัดหมวด",
    "semantic": "แยกทุกพิกเซลของภาพ (segmentation)",
    "ner": "ดึงข้อมูลสำคัญจากข้อความ (NER)",
    "docs": "จัดหมวดเอกสาร",
    "regress": "ทายตัวเลข (regression)",
    "forecast": "พยากรณ์อนาคตตามเวลา (time series)",
}


def _count(folder: Path, exts: set[str], limit: int = 200_000) -> int:
    n = 0
    for p in folder.rglob("*"):
        if p.suffix.lower() in exts:
            n += 1
            if n >= limit:
                break
    return n


def _thai_share(texts: list[str]) -> float:
    s = "".join(texts)[:20_000]
    return sum("\u0e00" <= ch <= "\u0e7f" for ch in s) / max(1, len(s))


def _gpu() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"gpu": True, "gpu_name": p.name, "vram_gb": round(p.total_memory / 2**30, 1)}
    except Exception:  # noqa: BLE001 - torch missing / broken driver: CPU
        pass
    return {"gpu": False, "gpu_name": None, "vram_gb": 0}


# --------------------------------------------------------------------------- tables
def _guess_target(df: pd.DataFrame) -> str:
    low = {c.lower().strip(): c for c in df.columns}
    for n in TARGET_NAMES:
        if n in low:
            return low[n]
    return df.columns[-1]


def _guess_time(df: pd.DataFrame) -> str | None:
    from ml_training.timeseries import find_time_column

    try:
        return find_time_column(df)
    except (ValueError, KeyError):
        return None


def _guess_group(df: pd.DataFrame, time_col: str | None, target: str) -> str | None:
    """A text column that repeats every time stamp (shop, product ...) = several series."""
    if not time_col:
        return None
    n_times = df[time_col].nunique()
    for c in df.columns:
        if c in (time_col, target) or not (
            pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
        ):
            continue
        k = df[c].nunique()
        if 1 < k <= 1000 and abs(len(df) / k - n_times) <= 0.2 * n_times:
            return c
    return None


def _long_text_column(df: pd.DataFrame) -> str | None:
    best, best_len = None, 0.0
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            avg = df[c].dropna().astype(str).head(500).str.len().mean()
            if avg and avg > best_len:
                best, best_len = c, avg
    named = next((c for c in df.columns if c.lower() in TEXT_NAMES), None)
    if named and (
        pd.api.types.is_string_dtype(df[named]) or pd.api.types.is_object_dtype(df[named])
    ):
        return named
    return best if best_len >= 25 else None


def inspect_table(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in CONLL_EXTS or _is_conll(path):
        return {"kind": "text", "task": "ner", "summary": "ไฟล์ CoNLL (คำ + ป้ายกำกับ) สำหรับ NER"}
    df = read_table(path)
    if {"tokens", "tags"} <= set(df.columns):
        return {
            "kind": "text",
            "task": "ner",
            "rows": len(df),
            "summary": f"ประโยคที่ติดป้ายแล้ว {len(df):,} ประโยค (tokens + tags)",
        }
    cols = [str(c) for c in df.columns]
    text_col = _long_text_column(df)
    if text_col:  # long free text + a label column = text classification
        label = next((c for c in df.columns if c.lower() in LABEL_NAMES and c != text_col), None)
        if label is None and len(df.columns) == 2:
            label = next(c for c in df.columns if c != text_col)
        if label is not None:
            texts = df[text_col].dropna().astype(str).head(300).tolist()
            classes = df[label].astype(str).value_counts()
            return {
                "kind": "text",
                "task": "classify",
                "rows": len(df),
                "columns": cols,
                "text_column": text_col,
                "label_column": str(label),
                "classes": classes.index.tolist()[:30],
                "n_classes": int(len(classes)),
                "thai": _thai_share(texts) > 0.3,
                "summary": f"ข้อความ {len(df):,} แถว (คอลัมน์ '{text_col}') ป้ายกำกับ '{label}' "
                f"{len(classes)} หมวด",
            }
    from ml_training.features import detect_task

    target = _guess_target(df)
    time_col = _guess_time(df)
    group = _guess_group(df, time_col, target)
    y = df[target].dropna()
    task = detect_task(y, OmegaConf.create({"task": "auto"}))
    if time_col and task == "regress":
        task = "forecast"
    out = {
        "kind": "table",
        "task": task,
        "rows": len(df),
        "columns": cols,
        "target": str(target),
        "time_column": time_col,
        "group_column": group,
        "missing_pct": round(float(df.isna().mean().mean() * 100), 1),
    }
    if task == "classify":
        out["classes"] = y.astype(str).value_counts().index.tolist()[:30]
        out["n_classes"] = int(y.nunique())
    detail = (
        f"{len(out.get('classes', []))} หมวด"
        if task == "classify"
        else f"ค่า {y.min():,.4g} – {y.max():,.4g}"
    )
    out["summary"] = f"ตาราง {len(df):,} แถว {len(cols)} คอลัมน์ · ทาย '{target}' ({detail})"
    if task == "forecast":
        out["summary"] += f" ตามเวลา '{time_col}'" + (f" แยกตาม '{group}'" if group else "")
    return out


# --------------------------------------------------------------------------- folders
def _looks_like_pose(root: Path) -> bool:
    """Pose rows all have the same width 5 + 3k (box + k keypoints); polygons vary row by row."""
    widths = set()
    for n, lbl in enumerate(p for p in root.rglob("*.txt") if "labels" in p.parts):
        if n >= 200:
            break
        widths |= {
            len(r.split()) for r in lbl.read_text(encoding="utf-8").splitlines() if r.strip()
        }
    return len(widths) == 1 and (w := widths.pop()) > 9 and ((w - 5) % 3 == 0 or (w - 5) % 2 == 0)


def inspect_folder(root: Path) -> dict[str, Any]:
    from ml_training.detection_data import _class_dirs, _classify_split_dirs, detect_task
    from ml_training.vision_data import detect_task as vision_task

    f = _input_file(root)
    if f is not None and f.is_file():
        return {**inspect_table(f), "file": str(f)}
    n_img = _count(root, IMG_EXTS)
    n_doc = _count(root, DOC_EXTS)
    n_lbl = sum(1 for p in root.rglob("*.txt") if "labels" in p.parts)
    has_yaml = any(root.glob("*.yaml"))
    if n_lbl or (has_yaml and n_img):
        task = detect_task(root, OmegaConf.create({}))
        if task == "segment" and _looks_like_pose(root):
            task = "pose"
        return {
            "kind": "yolo",
            "task": task,
            "images": n_img,
            "label_files": n_lbl,
            "summary": f"ภาพ {n_img:,} ภาพ พร้อมไฟล์ป้ายกำกับ {n_lbl:,} ไฟล์ (รูปแบบ YOLO)",
        }
    if n_img and vision_task(root) == "segment":
        return {
            "kind": "vision",
            "task": "semantic",
            "images": n_img,
            "summary": f"ภาพ {n_img:,} ไฟล์ (รวม mask) สำหรับแยกทุกพิกเซล",
        }
    splits = _classify_split_dirs(root)
    if splits:  # one sub-folder per class (optionally inside train / val / test)
        classes = sorted({c.name for d in splits.values() for c in _class_dirs(d)})
        return {
            "kind": "vision",
            "task": "classify",
            "images": n_img,
            "classes": classes[:50],
            "n_classes": len(classes),
            "summary": f"ภาพ {n_img:,} ภาพ แยกโฟลเดอร์ตาม {len(classes)} หมวด: "
            + ", ".join(classes[:8])
            + (" ..." if len(classes) > 8 else ""),
        }
    sub = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if n_doc and sub:  # documents per class
        classes = sorted(d.name for d in sub if _count(d, DOC_EXTS | IMG_EXTS, 1))
        if len(classes) >= 2:
            sample = [
                p.read_text(encoding="utf-8", errors="ignore")[:500]
                for p in list(root.rglob("*.txt"))[:50]
            ]
            return {
                "kind": "text",
                "task": "docs",
                "documents": n_doc,
                "classes": classes,
                "n_classes": len(classes),
                "thai": _thai_share(sample) > 0.3 if sample else True,
                "summary": f"เอกสาร {n_doc:,} ไฟล์ แยกโฟลเดอร์ตาม {len(classes)} หมวด",
            }
    if n_img:
        raise ValueError(
            f"พบภาพ {n_img:,} ภาพ แต่ยังไม่มีป้ายกำกับ\n"
            "  - จัดหมวดภาพ: แยกภาพใส่โฟลเดอร์ย่อยตามหมวด เช่น cat\\  dog\\\n"
            "  - ตรวจจับวัตถุ: ตีกรอบด้วย Roboflow / CVAT / Label Studio แล้ว export แบบ YOLO"
        )
    raise ValueError("ไม่พบข้อมูลที่ใช้เทรนได้ในโฟลเดอร์นี้ (ภาพ เอกสาร หรือไฟล์ CSV / Excel)")


def inspect(path: str | Path) -> dict[str, Any]:
    p = Path(str(path).strip().strip('"'))
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์หรือโฟลเดอร์: {p}")
    if p.is_file() and zipfile.is_zipfile(p) and p.suffix.lower() == ".zip":
        raise ValueError(f"{p.name} เป็นไฟล์ zip – แตกไฟล์ (คลิกขวา > Extract All) แล้วลากโฟลเดอร์มาแทน")
    if p.is_file():
        if p.suffix.lower() not in TABLE_EXTS | CONLL_EXTS:
            raise ValueError(f"ยังไม่รองรับไฟล์ {p.suffix} – ใช้ CSV / Excel / JSONL หรือทั้งโฟลเดอร์")
        info = inspect_table(p)
    else:
        info = inspect_folder(p)
    name = p.stem if p.is_file() else p.name
    info = {"path": str(p.resolve()), "name": name.replace(" ", "_"), **info, **_gpu()}
    info["task_th"] = TASK_TH.get(info["task"], info["task"])
    if info["kind"] == "text" and info["task"] == "docs":
        info["task"] = "classify"
    return info


def _counter_line(info: dict[str, Any]) -> str:
    gpu = (
        f"มี GPU {info['gpu_name']} ({info['vram_gb']} GB)"
        if info["gpu"]
        else "ไม่มี GPU (ใช้ CPU – ช้ากว่า)"
    )
    return f"{info['summary']}\nงานที่เหมาะ: {info['task_th']}\nเครื่องนี้: {gpu}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="What is this data and how to train it")
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        info = inspect(a.path)
    except (FileNotFoundError, ValueError, KeyError) as e:
        msg = str(e).strip("'\"")
        print(json.dumps({"error": msg}, ensure_ascii=False) if a.json else msg)
        return 2
    print(json.dumps(info, ensure_ascii=False, default=str) if a.json else _counter_line(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
