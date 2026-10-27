"""Matching predictions to ground truth + the statistics behind the XAI workbook.

Pure numpy / pandas / scipy (no torch) so every number can be unit-tested.

Error types (in the spirit of TIDE, Bolya et al. 2020):
  FP  background      – no object of any class there (IoU < 0.1 with every GT box)
      localization    – right class but the box is off (0.1 <= IoU < iou_thr)
      class           – an object is there but another class (IoU >= iou_thr, wrong class)
      class+loc       – wrong class and the box is off
      duplicate       – right class, right place, but that object was already found
  FN  missed          – nothing predicted near the object
      wrong class     – found at the right place but with another class
      poor box        – right class but the box overlaps too little
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

SIZE_BUCKETS = (("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, math.inf))


# --------------------------------------------------------------------------- ground truth
def load_gt(label_file: Path, task: str, width: int, height: int) -> list[tuple[int, list[float]]]:
    """YOLO label file -> [(class, [x1, y1, x2, y2] in pixels)] for detect/segment/pose/obb."""
    if not label_file.exists():
        return []
    out = []
    for row in label_file.read_text(encoding="utf-8").splitlines():
        v = row.split()
        if len(v) < 5:
            continue
        c, nums = int(float(v[0])), [float(x) for x in v[1:]]
        if task == "pose" or len(nums) == 4:  # cx cy w h (+ keypoints)
            cx, cy, w, h = nums[:4]
            box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        else:  # polygon (segment) or 4 corners (obb)
            xs, ys = nums[0::2], nums[1::2]
            box = [min(xs), min(ys), max(xs), max(ys)]
        out.append((c, [box[0] * width, box[1] * height, box[2] * width, box[3] * height]))
    return out


def size_bucket(box: Sequence[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return next(name for name, lo, hi in SIZE_BUCKETS if lo <= area < hi)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


# --------------------------------------------------------------------------- matching
def match_image(
    preds: Sequence[dict[str, Any]],
    gts: Sequence[tuple[int, Sequence[float]]],
    iou_thr: float = 0.5,
    bg_thr: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy matching by confidence. preds: dicts with cls, conf, box. Returns (pred rows, gt rows)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    p_boxes = np.array([preds[i]["box"] for i in order]).reshape(-1, 4)
    g_boxes = np.array([g[1] for g in gts]).reshape(-1, 4)
    g_cls = np.array([g[0] for g in gts], dtype=int)
    ious = iou_matrix(p_boxes, g_boxes)
    taken = np.zeros(len(gts), bool)
    p_rows: list[dict[str, Any]] = []
    for k, i in enumerate(order):
        p = preds[i]
        row = {
            **p,
            "outcome": "FP",
            "error": "background",
            "iou": 0.0,
            "gt_index": -1,
            "gt_cls": -1,
        }
        if len(gts):
            same = g_cls == p["cls"]
            iou_same = np.where(same & ~taken, ious[k], -1)
            j = int(iou_same.argmax())
            if iou_same[j] >= iou_thr:
                taken[j] = True
                row |= {"outcome": "TP", "error": "", "iou": float(ious[k, j]), "gt_index": j}
                row["gt_cls"] = int(g_cls[j])
            else:
                j_any = int(ious[k].argmax())
                best = float(ious[k, j_any])
                best_same = float(np.where(same, ious[k], -1).max()) if same.any() else -1.0
                row |= {"iou": best, "gt_index": j_any, "gt_cls": int(g_cls[j_any])}
                if best_same >= iou_thr:
                    row["error"] = "duplicate"
                elif best >= iou_thr:
                    row["error"] = "class"
                elif best_same >= bg_thr:
                    row["error"] = "localization"
                elif best >= bg_thr:
                    row["error"] = "class+loc"
        p_rows.append(row)
    g_rows = []
    for j, (c, box) in enumerate(gts):
        col = ious[:, j] if len(preds) else np.zeros(0)
        same = np.array([preds[i]["cls"] == c for i in order], bool)
        best_any = float(col.max()) if len(col) else 0.0
        best_same = float(col[same].max()) if same.any() else 0.0
        fn_type = ""
        if not taken[j]:
            if best_any >= iou_thr and best_same < iou_thr:
                fn_type = "wrong class"
            elif best_same >= bg_thr:
                fn_type = "poor box"
            else:
                fn_type = "missed"
        g_rows.append(
            {"cls": c, "box": list(box), "detected": bool(taken[j]), "fn_type": fn_type,
             "best_iou": max(best_any, 0.0), "size": size_bucket(box)}
        )  # fmt: skip
    return p_rows, g_rows


# --------------------------------------------------------------------------- attention
def box_attention(cam: np.ndarray, box: Sequence[float]) -> dict[str, float]:
    """How much of the heatmap falls inside a box.

    inside = share of all heat inside the box, area = box share of the image,
    lift = inside / area (1 = no focus, >1 = the model looks at the box),
    peak_in = 1 if the hottest pixel of the map is inside the box (pointing game).
    """
    h, w = cam.shape
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    total = float(cam.sum()) or 1e-9
    if x2 <= x1 or y2 <= y1:
        return {"inside": 0.0, "area": 0.0, "lift": 0.0, "peak_in": 0.0}
    inside = float(cam[y1:y2, x1:x2].sum()) / total
    area = (x2 - x1) * (y2 - y1) / (w * h)
    py, px = np.unravel_index(int(cam.argmax()), cam.shape)
    return {
        "inside": inside,
        "area": area,
        "lift": inside / area if area else 0.0,
        "peak_in": float(x1 <= px < x2 and y1 <= py < y2),
    }


def heat_spread(cam: np.ndarray) -> float:
    """Normalised entropy of the heatmap: 0 = one hot spot, 1 = heat everywhere."""
    p = cam.astype(float).ravel()
    p = p / (p.sum() or 1e-9)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / math.log(len(p))) if len(p) > 1 else 0.0


# --------------------------------------------------------------------------- statistics
def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    m = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def bootstrap_prf(
    per_image: pd.DataFrame, n_boot: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95% percentile CI of precision / recall / F1 by resampling images (columns TP, FP, FN)."""
    if per_image.empty or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    arr = per_image[["TP", "FP", "FN"]].to_numpy(float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    sums = arr[idx].sum(1)
    tp, fp, fn = sums[:, 0], sums[:, 1], sums[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0)
    q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))  # noqa: E731
    return {"precision": q(p), "recall": q(r), "f1": q(f)}


def calibration(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """Reliability table + ECE / MCE / Brier: does conf 0.8 really mean 80% correct?"""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "mean_conf": np.nan,
                         "accuracy": np.nan, "gap": np.nan})  # fmt: skip
            continue
        acc, mc = float(correct[m].mean()), float(conf[m].mean())
        gap = acc - mc
        ece += n / len(conf) * abs(gap)
        mce = max(mce, abs(gap))
        rows.append(
            {"bin": f"{lo:.1f}-{hi:.1f}", "n": n, "mean_conf": mc, "accuracy": acc, "gap": gap}
        )
    brier = float(((conf - correct) ** 2).mean()) if len(conf) else math.nan
    return pd.DataFrame(rows), {"ece": ece, "mce": mce, "brier": brier}


def threshold_sweep(
    images: Sequence[tuple[list[dict], list[tuple[int, list[float]]]]],
    class_names: Sequence[str],
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2)),
    iou_thr: float = 0.5,
) -> pd.DataFrame:
    """P / R / F1 for every confidence threshold, overall and per class."""
    rows = []
    for t in thresholds:
        tp = np.zeros(len(class_names))
        fp = np.zeros(len(class_names))
        fn = np.zeros(len(class_names))
        for preds, gts in images:
            pr, gr = match_image([p for p in preds if p["conf"] >= t], gts, iou_thr)
            for r in pr:
                (tp if r["outcome"] == "TP" else fp)[r["cls"]] += 1
            for g in gr:
                if not g["detected"]:
                    fn[g["cls"]] += 1
        p, r, f = prf(tp.sum(), fp.sum(), fn.sum())
        rows.append({"conf": t, "class": "ALL", "TP": tp.sum(), "FP": fp.sum(), "FN": fn.sum(),
                     "precision": p, "recall": r, "f1": f})  # fmt: skip
        for c, name in enumerate(class_names):
            if tp[c] + fp[c] + fn[c] == 0:
                continue
            p, r, f = prf(tp[c], fp[c], fn[c])
            rows.append({"conf": t, "class": name, "TP": tp[c], "FP": fp[c], "FN": fn[c],
                         "precision": p, "recall": r, "f1": f})  # fmt: skip
    return pd.DataFrame(rows)


def _row(name: str, question: str, n: str, stat: float, p: float, effect: float | None,
         effect_name: str, yes: str, no: str) -> dict[str, Any]:  # fmt: skip
    sig = bool(p < 0.05) if not math.isnan(p) else False
    return {
        "test": name,
        "question (ภาษาไทย)": question,
        "n": n,
        "statistic": stat,
        "p_value": p,
        "effect_size": effect,
        "effect": effect_name,
        "significant (p<0.05)": "yes" if sig else "no",
        "interpretation": yes if sig else no,
    }


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """U, p (two-sided) and rank-biserial effect size (+1 = a always larger)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return math.nan, math.nan, math.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p), float(2 * u / (len(a) * len(b)) - 1)


def hypothesis_tests(det: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """The questions people ask about a detector, answered with a test + effect size."""
    rows = []
    tp, fp = det[det.outcome == "TP"], det[det.outcome == "FP"]

    if "grad_lift" in det:
        u, p, r = mann_whitney(tp.grad_lift, fp.grad_lift)
        rows.append(_row(
            "Mann-Whitney U", "ตอนทายถูก (TP) โมเดลมองที่ตัววัตถุมากกว่าตอนทายผิด (FP) ไหม",
            f"TP={tp.grad_lift.notna().sum()} FP={fp.grad_lift.notna().sum()}", u, p, r,
            "rank-biserial (Grad-CAM lift)",
            "ใช่ – ความร้อน Grad-CAM อยู่ในกรอบของ TP มากกว่า FP อย่างมีนัยสำคัญ: เมื่อทายถูกโมเดลดูที่วัตถุจริง",
            "ไม่ต่างกันชัดเจน – FP ก็ถูกตัดสินจากบริเวณในกรอบพอๆ กับ TP (ดูภาพใน Gallery ว่า FP เหมือนวัตถุจริงไหม)",
        ))  # fmt: skip
    u, p, r = mann_whitney(tp.conf, fp.conf)
    rows.append(_row(
        "Mann-Whitney U", "ความมั่นใจ (conf) ของ TP สูงกว่า FP ไหม", f"TP={len(tp)} FP={len(fp)}",
        u, p, r, "rank-biserial (conf)",
        "ใช่ – conf แยกถูก/ผิดได้: ปรับ threshold ช่วยลด FP ได้จริง (ดูชีต Threshold)",
        "conf ของ TP และ FP ใกล้กัน – การปรับ threshold จะลด FP ได้น้อย ต้องแก้ที่ข้อมูล/โมเดล",
    ))  # fmt: skip
    if len(tp) >= 3:
        rho, p = stats.spearmanr(tp.conf, tp.iou)
        rows.append(_row(
            "Spearman correlation", "ยิ่งมั่นใจ กรอบยิ่งพอดีวัตถุไหม (conf vs IoU ของ TP)",
            str(len(tp)), float(rho), float(p), float(rho), "rho",
            "ใช่ – conf สะท้อนความแม่นของกรอบ", "conf ไม่ค่อยบอกว่ากรอบพอดีแค่ไหน",
        ))  # fmt: skip
    if "eigen_lift" in gt and gt.detected.any() and (~gt.detected).any():
        u, p, r = mann_whitney(gt[gt.detected].eigen_lift, gt[~gt.detected].eigen_lift)
        rows.append(_row(
            "Mann-Whitney U", "วัตถุที่หาไม่เจอ (FN) ได้รับความสนใจจากโมเดลน้อยกว่าวัตถุที่หาเจอไหม",
            f"found={int(gt.detected.sum())} missed={int((~gt.detected).sum())}", u, p, r,
            "rank-biserial (EigenCAM lift)",
            "ใช่ – โมเดลแทบไม่มองวัตถุที่พลาด: มักเล็ก/มืด/ถูกบัง ควรเพิ่มตัวอย่างแบบนี้ใน train",
            "โมเดลมองเห็นวัตถุที่พลาดพอๆ กัน แต่ตัดสินใจไม่ถูก – มักเป็นปัญหา class หรือ threshold",
        ))  # fmt: skip
    if gt["size"].nunique() > 1:
        tab = pd.crosstab(gt["size"], gt.detected)
        if tab.shape == (tab.shape[0], 2) and tab.shape[0] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(tab)
            v = math.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
            rows.append(_row(
                "Chi-square", "ขนาดวัตถุ (small/medium/large) มีผลต่อการหาเจอไหม",
                str(int(tab.values.sum())), float(chi2), float(p), v, "Cramer's V",
                "ใช่ – recall ต่างกันตามขนาดวัตถุ (ดูชีต Size): ลองเพิ่ม imgsz หรือเพิ่มภาพวัตถุขนาดนั้น",
                "ขนาดวัตถุไม่ใช่สาเหตุหลักของการพลาด",
            ))  # fmt: skip
        area = gt.box.apply(lambda b: max(1.0, (b[2] - b[0]) * (b[3] - b[1])))
        if gt.detected.nunique() == 2:
            r_pb, p = stats.pointbiserialr(gt.detected.astype(int), np.log(area))
            rows.append(_row(
                "Point-biserial correlation", "วัตถุยิ่งใหญ่ ยิ่งหาเจอง่ายไหม (log พื้นที่ vs หาเจอ)",
                str(len(gt)), float(r_pb), float(p), float(r_pb), "r",
                "ใช่ – ขนาดสัมพันธ์กับการหาเจอ", "ขนาดไม่สัมพันธ์กับการหาเจออย่างมีนัยสำคัญ",
            ))  # fmt: skip
    if len(fp) and fp.error.nunique() > 1 and fp["cls"].nunique() > 1:
        tab = pd.crosstab(fp["cls"], fp.error)
        chi2, p, _, _ = stats.chi2_contingency(tab)
        v = math.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
        rows.append(_row(
            "Chi-square", "ชนิดความผิดพลาด (FP) ขึ้นกับ class ไหม", str(len(fp)), float(chi2),
            float(p), v, "Cramer's V",
            "ใช่ – แต่ละ class ผิดคนละแบบ (ดูชีต Errors) แก้รายคลาสได้ตรงจุด",
            "ทุก class ผิดคล้ายๆ กัน",
        ))  # fmt: skip
    if len(fp) and fp.error.nunique() > 1:
        groups = [g.conf.to_numpy() for _, g in fp.groupby("error") if len(g) >= 2]
        if len(groups) > 1:
            h, p = stats.kruskal(*groups)
            rows.append(_row(
                "Kruskal-Wallis", "FP แต่ละชนิดมี conf ต่างกันไหม", str(len(fp)), float(h),
                float(p), None, "",
                "ใช่ – บางชนิดของ FP ตัดทิ้งได้ด้วย threshold", "conf ของ FP ทุกชนิดใกล้เคียงกัน",
            ))  # fmt: skip
    return pd.DataFrame(rows)


def per_class_table(
    det: pd.DataFrame, gt: pd.DataFrame, class_names: Sequence[str], sweep: pd.DataFrame | None
) -> pd.DataFrame:
    rows = []
    for c, name in enumerate(class_names):
        d, g = det[det["cls"] == c], gt[gt["cls"] == c]
        if d.empty and g.empty:
            continue
        tp = int((d.outcome == "TP").sum())
        fp = int((d.outcome == "FP").sum())
        fn = int((~g.detected).sum()) if len(g) else 0
        p, r, f = prf(tp, fp, fn)
        pl, ph = wilson(tp, tp + fp)
        rl, rh = wilson(tp, tp + fn)
        row: dict[str, Any] = {
            "class": name, "GT objects": len(g), "predictions": len(d), "TP": tp, "FP": fp, "FN": fn,
            "precision": p, "precision 95% low": pl, "precision 95% high": ph,
            "recall": r, "recall 95% low": rl, "recall 95% high": rh, "f1": f,
        }  # fmt: skip
        for e in ("background", "localization", "class", "class+loc", "duplicate"):
            row[f"FP {e}"] = int((d.error == e).sum())
        for e in ("missed", "wrong class", "poor box"):
            row[f"FN {e}"] = int((g.fn_type == e).sum()) if len(g) else 0
        row["mean conf TP"] = d[d.outcome == "TP"].conf.mean()
        row["mean conf FP"] = d[d.outcome == "FP"].conf.mean()
        row["mean IoU TP"] = d[d.outcome == "TP"].iou.mean()
        if "grad_lift" in d:
            row["attention lift TP"] = d[d.outcome == "TP"].grad_lift.mean()
            row["attention lift FP"] = d[d.outcome == "FP"].grad_lift.mean()
            row["pointing game TP"] = d[d.outcome == "TP"].grad_peak_in.mean()
        for s, _, _ in SIZE_BUCKETS:
            gs = g[g["size"] == s] if len(g) else g
            row[f"recall {s}"] = gs.detected.mean() if len(gs) else np.nan
            row[f"n {s}"] = len(gs)
        if sweep is not None and not sweep.empty:
            sc = sweep[sweep["class"] == name]
            if len(sc):
                best = sc.loc[sc.f1.idxmax()]
                row["best conf (max F1)"] = best.conf
                row["F1 at best conf"] = best.f1
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- classify
def coverage_sweep(conf: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """Selective prediction: answer only when conf >= t -> how many answered, how many right."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    rows = []
    for t in np.round(np.arange(0.0, 0.96, 0.05), 2):
        m = conf >= t
        rows.append({"conf": float(t), "class": "ALL", "answered": int(m.sum()),
                     "coverage": float(m.mean()) if len(m) else 0.0,
                     "accuracy": float(correct[m].mean()) if m.any() else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def classify_per_class(df: pd.DataFrame, class_names: Sequence[str]) -> pd.DataFrame:
    """df columns: true, pred (class indices), conf, correct, grad_spread."""
    rows = []
    for c, name in enumerate(class_names):
        t, p = df[df.true == c], df[df.pred == c]
        if t.empty and p.empty:
            continue
        tp = int(((df.true == c) & (df.pred == c)).sum())
        fp, fn = len(p) - tp, len(t) - tp
        pr, rc, f1 = prf(tp, fp, fn)
        rl, rh = wilson(tp, tp + fn)
        pl, ph = wilson(tp, tp + fp)
        wrong = t[t.pred != c]
        confused = wrong.pred.value_counts()
        rows.append({
            "class": name, "images": len(t), "correct": tp, "TP": tp, "FP": fp, "FN": fn,
            "precision": pr, "precision 95% low": pl, "precision 95% high": ph,
            "recall": rc, "recall 95% low": rl, "recall 95% high": rh, "f1": f1,
            "mean conf correct": t[t.correct].conf.mean(),
            "mean conf wrong": wrong.conf.mean(),
            "most confused with": class_names[int(confused.index[0])] if len(confused) else "",
            "times": int(confused.iloc[0]) if len(confused) else 0,
            "heat spread": t.grad_spread.mean() if "grad_spread" in t else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


def classify_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Statistical questions for a classifier (df: correct, conf, margin, entropy, grad_spread)."""
    rows = []
    ok, bad = df[df.correct], df[~df.correct]
    checks = [
        ("conf", "ตอนทายถูก โมเดลมั่นใจ (conf) มากกว่าตอนทายผิดไหม",
         "ใช่ – conf แยกถูก/ผิดได้: ตั้งเกณฑ์ conf เพื่อส่งภาพที่ไม่แน่ใจให้คนตรวจได้ (ดูชีต Threshold)",
         "conf ตอนถูกและผิดใกล้กัน – เชื่อ conf ไม่ได้มาก"),
        ("margin", "ช่องว่างระหว่างอันดับ 1 กับอันดับ 2 (margin) ของภาพที่ถูกกว้างกว่าไหม",
         "ใช่ – ภาพที่ผิดมักลังเลระหว่าง 2 class (ดู 'most confused with' ในชีต PerClass)",
         "margin ไม่ต่างกันชัดเจน"),
        ("entropy", "ภาพที่ทายผิด การกระจายความน่าจะเป็นสับสน (entropy สูง) กว่าไหม",
         "ใช่ – ภาพที่ผิดโมเดลไม่แน่ใจจริงๆ ไม่ใช่มั่นใจผิด",
         "ภาพที่ผิดก็ไม่ได้สับสนกว่า – โมเดล 'มั่นใจผิด' ควรตรวจ label / ภาพคล้ายกัน"),
        ("grad_spread", "Heatmap ของภาพที่ทายผิดกระจายกว่า (ไม่ได้โฟกัสจุดสำคัญ) ไหม",
         "ใช่ – ตอนผิดโมเดลมองกว้างไม่จับจุด: ภาพอาจมีพื้นหลังรก/วัตถุเล็ก",
         "การโฟกัสของ heatmap ไม่ต่างกันระหว่างถูกและผิด"),
    ]  # fmt: skip
    for col, q, yes, no in checks:
        if col not in df:
            continue
        u, p, r = mann_whitney(ok[col], bad[col])
        rows.append(_row("Mann-Whitney U", q, f"correct={len(ok)} wrong={len(bad)}", u, p, r,
                         f"rank-biserial ({col})", yes, no))  # fmt: skip
    if df.true.nunique() > 1 and df.correct.nunique() == 2:
        tab = pd.crosstab(df.true, df.correct)
        chi2, p, _, _ = stats.chi2_contingency(tab)
        v = math.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
        rows.append(_row("Chi-square", "ความแม่นยำต่างกันระหว่าง class อย่างมีนัยสำคัญไหม",
                         str(len(df)), float(chi2), float(p), v, "Cramer's V",
                         "ใช่ – บาง class ยากกว่าชัดเจน (ดูชีต PerClass) เพิ่มภาพของ class นั้น",
                         "ทุก class ยากพอๆ กัน"))  # fmt: skip
    return pd.DataFrame(rows)
