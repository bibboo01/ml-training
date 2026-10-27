"""Explain a trained YOLO model: heatmaps + why each detection was made + a statistics workbook.

uv run python -m ml_training.xai.run data=coco8 model=yolo11n
uv run python -m ml_training.xai.run data=helmet model=yolo11s xai.max_images=300 xai.split=val
uv run python -m ml_training.xai.run data=helmet model=yolo11s +weights=D:/best.pt

Writes reports/<dataset>/<model>/xai/:
  xai_report.xlsx   Summary, Stats, PerClass, Detections, Objects, Images, Errors, Size,
                    Calibration, Threshold, Gallery, Glossary
  images/           per image: boxes (green TP / red FP / orange FN) | EigenCAM | Grad-CAM
  boxes/            Grad-CAM crop of every analysed prediction
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ml_training import lineage
from ml_training.utils import CONFIG_DIR, log, read_json
from ml_training.xai import analysis as A
from ml_training.xai.cam import YoloExplainer, overlay
from ml_training.xai.report import write_xai_report

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GREEN, RED, ORANGE = (60, 180, 60), (40, 40, 230), (0, 140, 255)
DEFAULTS = {"split": "auto", "max_images": 100, "conf": 0.25, "iou": 0.5, "low_conf": 0.01,
            "max_boxes": 20, "bootstrap": 1000, "imgsz": None, "gallery": 12, "seed": 0}  # fmt: skip


def settings(cfg: DictConfig) -> dict[str, Any]:
    return DEFAULTS | dict(cfg.get("xai") or {})


# --------------------------------------------------------------------------- inputs
def pick_images(processed: Path, task: str, split: str, n: int, seed: int) -> tuple[str, list]:
    """Images of the chosen split (auto = test, else val). Classify -> (path, class folder)."""
    order = ["test", "val", "train"] if split == "auto" else [split]
    for s in order:
        if task == "classify":
            d = processed / s
            items = sorted((p, p.parent.name) for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)
        else:
            lst = processed / f"{s}.txt"
            text = lst.read_text(encoding="utf-8") if lst.exists() else ""
            items = [Path(x.strip()) for x in text.splitlines() if x.strip()]
        if items:
            if n and len(items) > n:
                items = sorted(random.Random(seed).sample(items, n))
            return s, items
    raise FileNotFoundError(
        f"No images for split '{split}' in {processed} – run prepare/featurize first"
    )


def find_weights(cfg: DictConfig) -> Path:
    w = Path(cfg.get("weights") or Path(cfg.paths.run_model).with_suffix(".pt"))
    if not w.exists():
        raise FileNotFoundError(
            f"{w} not found – train first (.\\run.ps1 fit ...) or pass +weights="
        )
    return w


# --------------------------------------------------------------------------- drawing
def _panel(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_NEAREST
                      if h < height / 2 else cv2.INTER_LINEAR)  # fmt: skip


def _label(img: np.ndarray, text: str, xy: tuple[int, int], color) -> None:
    s = max(0.4, min(img.shape[:2]) / 900)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
    x, y = max(0, xy[0]), max(th + 4, xy[1])
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
    cv2.putText(
        img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA
    )


def composite(panels: list[tuple[str, np.ndarray]], path: Path) -> Path:
    h0 = panels[0][1].shape[0]
    height = int(np.clip(h0, 320, 640))
    row = []
    for title, p in panels:
        p = _panel(p, height).copy()
        _label(p, title, (0, 22), (30, 30, 30))
        row.append(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def draw_boxes(img: np.ndarray, det: list[dict], gt: list[dict], names: list[str]) -> np.ndarray:
    out = img.copy()
    t = max(1, round(min(img.shape[:2]) / 300))
    for g in gt:
        if not g["detected"]:
            b = [int(v) for v in g["box"]]
            cv2.rectangle(out, b[:2], b[2:], ORANGE, t)
            _label(out, f"FN {names[g['cls']]} ({g['fn_type']})", (b[0], b[3] + 16), ORANGE)
    for d in det:
        b = [int(v) for v in d["box"]]
        color = GREEN if d["outcome"] == "TP" else RED
        cv2.rectangle(out, b[:2], b[2:], color, t)
        tag = (
            names[d["cls"]]
            + f" {d['conf']:.2f}"
            + ("" if d["outcome"] == "TP" else f" FP:{d['error']}")
        )
        _label(out, tag, (b[0], b[1]), color)
    return out


def crop(img: np.ndarray, box: list[float], path: Path, size: int = 256) -> Path:
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    m = 0.25 * max(bw, bh)
    x1, y1 = int(max(0, box[0] - m)), int(max(0, box[1] - m))
    x2, y2 = int(min(w, box[2] + m)), int(min(h, box[3] + m))
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        c = img
    s = size / max(c.shape[:2])
    c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))))
    cv2.rectangle(c, (round((box[0] - x1) * s), round((box[1] - y1) * s)),
                  (round((box[2] - x1) * s), round((box[3] - y1) * s)), (255, 255, 255), 1)  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), c)
    return path


# --------------------------------------------------------------------------- detect-type tasks
def _preds(r, task: str) -> list[dict]:
    src = r.obb if task == "obb" else r.boxes
    if src is None or len(src) == 0:
        return []
    xyxy = src.xyxy.cpu().numpy()
    cls = src.cls.cpu().numpy().astype(int)
    conf = src.conf.cpu().numpy()
    return [{"cls": int(c), "conf": float(p), "box": [float(v) for v in b]}
            for b, c, p in zip(xyxy, cls, conf, strict=True)]  # fmt: skip


def explain_detection(yolo, task: str, items: list[Path], names: list[str], x: dict, out: Path):
    from ml_training.detection_data import img2label

    explainer = None
    det_rows, gt_rows, img_rows, sweep_imgs, calib_rows = [], [], [], [], []
    for n, path in enumerate(items, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        r = yolo.predict(str(path), conf=x["low_conf"], imgsz=x["imgsz"], verbose=False)[0]
        if explainer is None:  # after the first predict the model is fused and on its device
            explainer = YoloExplainer(yolo, task, x["imgsz"])
        preds = _preds(r, task)
        gts = A.load_gt(img2label(path), task, w, h)
        sweep_imgs.append((preds, gts))
        calib_rows += A.match_image(preds, gts, x["iou"])[0]
        drows, grows = A.match_image([p for p in preds if p["conf"] >= x["conf"]], gts, x["iou"])

        explainer.run(img)
        eig = explainer.eigencam()
        union = np.zeros_like(eig)
        for k, d in enumerate(drows):
            d |= {"image": path.name, "class": names[d["cls"]], "size": A.size_bucket(d["box"]),
                  "gt_class": names[d["gt_cls"]] if d["gt_cls"] >= 0 else ""}  # fmt: skip
            e = A.box_attention(eig, d["box"])
            d |= {
                "eigen_inside": e["inside"],
                "eigen_lift": e["lift"],
                "eigen_peak_in": e["peak_in"],
            }
            if k < x["max_boxes"]:
                score, level = explainer.box_score(d["box"], d["cls"])
                if score is not None:
                    cam = explainer.gradcam(score, level)
                    union = np.maximum(union, cam)
                    g = A.box_attention(cam, d["box"])
                    d |= {"grad_inside": g["inside"], "grad_lift": g["lift"],
                          "grad_peak_in": g["peak_in"], "grad_spread": A.heat_spread(cam),
                          "heatmap": str(crop(overlay(img, cam), d["box"],
                                              out / "boxes" / f"{path.stem}_{k:02d}.jpg"))}  # fmt: skip
        for j, g in enumerate(grows):
            g |= {"image": path.name, "class": names[g["cls"]], "gt_index": j,
                  "eigen_lift": A.box_attention(eig, g["box"])["lift"]}  # fmt: skip
        tp = sum(d["outcome"] == "TP" for d in drows)
        fp, fn = len(drows) - tp, sum(not g["detected"] for g in grows)
        view = composite(
            [("boxes: green TP / red FP / orange FN", draw_boxes(img, drows, grows, names)),
             ("EigenCAM (where the model looks)", overlay(img, eig)),
             ("Grad-CAM (why each box)", overlay(img, union))],
            out / "images" / f"{path.stem}.jpg",
        )  # fmt: skip
        lifts = [d["grad_lift"] for d in drows if "grad_lift" in d]
        img_rows.append({
            "image": path.name, "objects": len(grows), "predictions": len(drows), "TP": tp, "FP": fp,
            "FN": fn, "errors": fp + fn, "mean grad lift": float(np.mean(lifts)) if lifts else np.nan,
            "eigen spread": A.heat_spread(eig), "view": str(view), "path": str(path),
        })  # fmt: skip
        det_rows += drows
        gt_rows += grows
        if n % 10 == 0 or n == len(items):
            log.info("XAI %d/%d images", n, len(items))
    if explainer:
        explainer.close()

    det_cols = ["image", "class", "conf", "outcome", "error", "iou", "gt_class", "size", "grad_lift",
                "grad_inside", "grad_peak_in", "grad_spread", "eigen_lift", "eigen_inside", "box",
                "cls", "heatmap"]  # fmt: skip
    det = pd.DataFrame(det_rows).reindex(columns=det_cols)
    gt = pd.DataFrame(gt_rows).reindex(
        columns=[
            "image",
            "class",
            "detected",
            "fn_type",
            "best_iou",
            "size",
            "eigen_lift",
            "box",
            "cls",
        ]
    )
    det["cls"] = det["cls"].fillna(-1).astype(int)
    gt["cls"] = gt["cls"].fillna(-1).astype(int)
    gt["detected"] = gt["detected"].astype(bool)
    images = pd.DataFrame(img_rows)
    return det, gt, images, sweep_imgs, pd.DataFrame(calib_rows)


def detection_tables(det, gt, images, sweep_imgs, calib_rows, names, x):
    tests = A.hypothesis_tests(det, gt)
    sweep = A.threshold_sweep(sweep_imgs, names, iou_thr=x["iou"])
    per_class = A.per_class_table(det, gt, names, sweep)
    correct = (calib_rows.outcome == "TP").to_numpy() if len(calib_rows) else np.zeros(0)
    conf = calib_rows.conf.to_numpy() if len(calib_rows) else np.zeros(0)
    calib, cal = A.calibration(conf, correct)
    boot = A.bootstrap_prf(images, x["bootstrap"]) if len(images) else {}
    fp_types = ["background", "localization", "class", "class+loc", "duplicate"]
    fn_types = ["missed", "wrong class", "poor box"]
    errors = None
    if len(per_class):
        err = per_class[["class"] + [f"FP {e}" for e in fp_types] + [f"FN {e}" for e in fn_types]]
        errors = err[err.iloc[:, 1:].sum(axis=1) > 0]
    size_rows = []
    for s, _, _ in A.SIZE_BUCKETS:
        g = gt[gt["size"] == s]
        if g.empty:
            continue
        k = int(g.detected.sum())
        lo, hi = A.wilson(k, len(g))
        size_rows.append({"size": s, "objects": len(g), "found": k, "recall": k / len(g),
                          "95% low": lo, "95% high": hi, "mean EigenCAM lift": g.eigen_lift.mean(),
                          "missed": int((g.fn_type == "missed").sum())})  # fmt: skip
    size = pd.DataFrame(size_rows)

    tp, fp = det[det.outcome == "TP"], det[det.outcome == "FP"]
    n_tp, n_fp, n_fn = len(tp), len(fp), int((~gt.detected).sum())
    p, r, f1 = A.prf(n_tp, n_fp, n_fn)
    ci = lambda k: f"{boot[k][0]:.3f} – {boot[k][1]:.3f}" if k in boot else "-"  # noqa: E731
    all_sweep = sweep[sweep["class"] == "ALL"]
    best = all_sweep.loc[all_sweep.f1.idxmax()] if len(all_sweep) else None
    top_fp = fp.error.value_counts()
    top_fn = gt[~gt.detected].fn_type.value_counts()
    headline = [
        ("images analysed", len(images), ""),
        ("objects (ground truth)", len(gt), ""),
        (f"predictions at conf ≥ {x['conf']}", len(det), ""),
        ("TP / FP / FN", f"{n_tp} / {n_fp} / {n_fn}", "ถูก / ทายเกิน / พลาด"),
        ("precision", p, f"95% CI (bootstrap): {ci('precision')} – ทายว่ามี แล้วมีจริงกี่ %"),
        ("recall", r, f"95% CI (bootstrap): {ci('recall')} – ของจริงทั้งหมด หาเจอกี่ %"),
        ("F1", f1, f"95% CI (bootstrap): {ci('f1')}"),
        ("best conf threshold (max F1)", float(best.conf) if best is not None else None,
         f"F1 = {best.f1:.3f} ที่ conf นี้ (ตอนนี้ใช้ {x['conf']}) – ดูชีต Threshold"
         if best is not None else ""),
        ("most common FP", f"{top_fp.index[0]} ({top_fp.iloc[0]})" if len(top_fp) else "-",
         "ชนิดของการทายเกินที่พบบ่อยที่สุด (ดู Glossary)"),
        ("most common FN", f"{top_fn.index[0]} ({top_fn.iloc[0]})" if len(top_fn) else "-",
         "สาเหตุที่พลาดบ่อยที่สุด"),
        ("pointing game (TP)", tp.grad_peak_in.mean() if len(tp) else None,
         "% ของกรอบที่ถูก ซึ่งจุดร้อนสุดของ Grad-CAM อยู่ในกรอบ – สูง = โมเดลตัดสินจากตัววัตถุจริง"),
        ("attention lift TP", tp.grad_lift.mean() if len(tp) else None,
         ">1 = มองที่วัตถุ · <1 = ใช้บริบทรอบๆ ตัดสิน"),
        ("attention lift FP", fp.grad_lift.mean() if len(fp) else None,
         "ถ้าใกล้ TP = FP หน้าตาเหมือนวัตถุจริง · ถ้าต่ำ = หลงกับพื้นหลัง/บริบท"),
        ("ECE (calibration)", cal["ece"], "ยิ่งต่ำยิ่งเชื่อค่า conf ได้ (ดูชีต Calibration)"),
        ("Brier score", cal["brier"], "0 = conf สมบูรณ์แบบ"),
    ]  # fmt: skip
    for _, s in size.iterrows():
        headline.append((f"recall {s['size']} objects", s["recall"], f"{s['found']}/{s['objects']} · "
                         f"95% CI {s['95% low']:.2f}-{s['95% high']:.2f}"))  # fmt: skip
    metrics = {"precision": p, "recall": r, "f1": f1, "ece": cal["ece"], "brier": cal["brier"],
               "TP": n_tp, "FP": n_fp, "FN": n_fn}  # fmt: skip
    if len(tp):
        metrics["pointing_game_tp"] = float(tp.grad_peak_in.mean())
    show = det.drop(columns=["cls"]).assign(box=det.box.map(_box_str))
    objs = gt.drop(columns=["cls"]).assign(box=gt.box.map(_box_str))
    sheets = [("PerClass", per_class, ""), ("Detections", show, "heatmap"),
              ("Objects", objs, ""), ("Images", images.drop(columns=["path"]), "view")]  # fmt: skip
    gallery = _gallery(images, x["gallery"], "errors", "FP {FP} · FN {FN} · TP {TP}")
    return dict(tests=tests, sheets=sheets, calib=calib, sweep=sweep, errors=errors, size=size,
                headline=headline, metrics=metrics, gallery=gallery)  # fmt: skip


def _box_str(b) -> str:
    return ", ".join(f"{v:.0f}" for v in b) if isinstance(b, list | tuple) else ""


def _gallery(images: pd.DataFrame, n: int, sort_col: str, fmt: str) -> list[tuple[str, Path]]:
    if images.empty or n <= 0:
        return []
    worst = images.sort_values(sort_col, ascending=False).head(max(1, n - n // 3))
    best = images.drop(worst.index).sort_values(sort_col).head(n // 3)
    out = [
        (f"WORST · {r.image} · " + fmt.format(**r._asdict()), Path(r.view))
        for r in worst.itertuples()
    ]
    out += [
        (f"BEST · {r.image} · " + fmt.format(**r._asdict()), Path(r.view))
        for r in best.itertuples()
    ]
    return out


# --------------------------------------------------------------------------- classify
def explain_classify(yolo, items: list[tuple[Path, str]], names: list[str], x: dict, out: Path):
    explainer = None
    rows = []
    for n, (path, folder) in enumerate(items, 1):
        img = cv2.imread(str(path))
        if img is None or folder not in names:
            continue
        r = yolo.predict(str(path), imgsz=x["imgsz"], verbose=False)[0]
        if explainer is None:
            explainer = YoloExplainer(yolo, "classify", x["imgsz"])
        probs = r.probs.data.cpu().numpy().astype(float)
        true, top = names.index(folder), np.argsort(-probs)
        pred = int(top[0])
        explainer.run(img)
        eig = explainer.eigencam()
        cam = explainer.gradcam(explainer.class_score(pred))
        panels = [
            (f"true {folder} / pred {names[pred]} {probs[pred]:.2f}", img),
            ("EigenCAM", overlay(img, eig)),
            (f"Grad-CAM '{names[pred]}'", overlay(img, cam)),
        ]
        if pred != true:
            panels.append(
                (
                    f"Grad-CAM '{folder}' (true)",
                    overlay(img, explainer.gradcam(explainer.class_score(true))),
                )
            )
        p = np.clip(probs, 1e-12, 1)
        rows.append({
            "image": path.name, "true_cls": folder, "pred_cls": names[pred], "conf": probs[pred],
            "correct": pred == true, "prob of true class": probs[true],
            "margin": probs[pred] - probs[top[1]] if len(top) > 1 else 1.0,
            "entropy": float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0,
            "second": names[int(top[1])] if len(top) > 1 else "",
            "grad_spread": A.heat_spread(cam), "eigen_spread": A.heat_spread(eig),
            "view": str(composite(panels, out / "images" / f"{folder}_{path.stem}.jpg")),
            "true": true, "pred": pred,
        })  # fmt: skip
        if n % 20 == 0 or n == len(items):
            log.info("XAI %d/%d images", n, len(items))
    if explainer:
        explainer.close()
    return pd.DataFrame(rows)


def classify_tables(df: pd.DataFrame, names: list[str], x: dict) -> dict:
    tests = A.classify_tests(df)
    per_class = A.classify_per_class(df, names)
    calib, cal = A.calibration(df.conf.to_numpy(), df.correct.to_numpy())
    sweep = A.coverage_sweep(df.conf.to_numpy(), df.correct.to_numpy())
    acc = float(df.correct.mean())
    lo, hi = A.wilson(int(df.correct.sum()), len(df))
    boot = (
        np.random.default_rng(0)
        .choice(df.correct.to_numpy(float), (x["bootstrap"], len(df)))
        .mean(1)
    )
    cm = pd.crosstab(df.true_cls, df.pred_cls).reindex(index=names, columns=names, fill_value=0)
    cm = cm.loc[cm.sum(axis=1) > 0].reset_index().rename(columns={"true_cls": "true \\ pred"})
    wrong = df[~df.correct]
    pairs = wrong.groupby(["true_cls", "pred_cls"]).size().sort_values(ascending=False)
    over = df[(~df.correct) & (df.conf >= 0.8)]
    headline = [
        ("images analysed", len(df), ""),
        ("accuracy (top1)", acc, f"95% CI Wilson {lo:.3f}-{hi:.3f} · bootstrap "
         f"{np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f}"),
        ("wrong images", len(wrong), "ดูภาพใน Gallery: heatmap ช่องขวาสุด = โมเดลมองตรงไหนของ class ที่ถูก"),
        ("most confused pair", f"{pairs.index[0][0]} → {pairs.index[0][1]} ({pairs.iloc[0]})" if len(pairs) else "-",
         "class จริง → class ที่โมเดลทาย ที่สับสนบ่อยที่สุด"),
        ("confident mistakes (conf ≥ 0.8)", len(over), "ผิดทั้งที่มั่นใจมาก – มักเป็น label ผิดหรือภาพกำกวม ควรตรวจ"),
        ("mean conf correct", df[df.correct].conf.mean(), ""),
        ("mean conf wrong", wrong.conf.mean() if len(wrong) else None, ""),
        ("ECE (calibration)", cal["ece"], "ยิ่งต่ำยิ่งเชื่อค่า conf ได้"),
        ("Brier score", cal["brier"], "0 = สมบูรณ์แบบ"),
    ]  # fmt: skip
    df = df.assign(errors=(~df.correct).astype(int) * (1 + df.conf))
    gallery = _gallery(df, x["gallery"], "errors", "true {true_cls} → pred {pred_cls} ({conf:.2f})")
    show = df.drop(columns=["true", "pred", "errors"])
    return dict(tests=tests, sheets=[("PerClass", per_class, ""), ("Confusion", cm, ""),
                                     ("Images", show, "view")],
                calib=calib, sweep=sweep, errors=None, size=None, headline=headline,
                metrics={"accuracy": acc, "ece": cal["ece"], "brier": cal["brier"]}, gallery=gallery)  # fmt: skip


# --------------------------------------------------------------------------- main
def explain(cfg: DictConfig) -> Path:
    from ultralytics import YOLO

    from ml_training.detection_data import configure_ultralytics

    configure_ultralytics(cfg)
    x = settings(cfg)
    processed = Path(cfg.paths.processed)
    meta = read_json(processed / "meta.json")
    task, names = meta["task"], list(meta["class_names"])
    weights = find_weights(cfg)
    yolo = YOLO(str(weights))
    x["imgsz"] = int(
        x["imgsz"] or yolo.overrides.get("imgsz") or (224 if task == "classify" else 640)
    )
    split, items = pick_images(processed, task, x["split"], int(x["max_images"]), int(x["seed"]))
    out = Path(cfg.paths.run_reports) / "xai"
    log.info("XAI: %s %s on %d %s images (task %s, imgsz %d)", cfg.data.name, cfg.model.name,
             len(items), split, task, x["imgsz"])  # fmt: skip

    if task == "classify":
        df = explain_classify(yolo, items, names, x, out)
        t = classify_tables(df, names, x)
    else:
        parts = explain_detection(yolo, task, items, names, x, out)
        t = detection_tables(*parts, names, x)

    git = lineage.git_info()
    info = {
        "dataset": cfg.data.name, "model": cfg.model.name, "task": task, "weights": str(weights),
        "weights sha256": lineage.fingerprint([weights])["sha256"][:16], "split": split,
        "images": len(items), "conf threshold": x["conf"], "IoU threshold": x["iou"],
        "image size": x["imgsz"], "Grad-CAM boxes per image": x["max_boxes"],
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "git commit": git.get("commit", "")[:10] if git.get("available") else "no git",
    }  # fmt: skip
    path = write_xai_report(out / "xai_report.xlsx", info, t["headline"], t["tests"], t["sheets"],
                            t["calib"], t["sweep"], t["errors"], t["size"], t["gallery"])  # fmt: skip
    log.info("XAI report: %s", path)
    track(cfg, info, t, path)
    return path


def track(cfg: DictConfig, info: dict, t: dict, path: Path) -> None:
    """History: an MLflow run (stage=xai) + a logs/history.jsonl line."""
    run_id = None
    metrics = {k: float(v) for k, v in t["metrics"].items() if v is not None and np.isfinite(v)}
    try:
        import mlflow

        from ml_training.tracking import setup_mlflow

        exp = setup_mlflow(cfg)
        with mlflow.start_run(experiment_id=exp, run_name=f"xai-{cfg.model.name}-"
                              f"{pd.Timestamp.now():%Y%m%d-%H%M%S}") as run:  # fmt: skip
            mlflow.set_tags({"stage": "xai", "dataset": cfg.data.name, "model": cfg.model.name})
            mlflow.log_params({k: str(v)[:500] for k, v in info.items()})
            mlflow.log_metrics({f"xai_{k}": v for k, v in metrics.items()})
            lineage.log_context()
            mlflow.log_artifact(str(path), artifact_path="xai")
            for _, img in t["gallery"]:
                if Path(img).exists():
                    mlflow.log_artifact(str(img), artifact_path="xai/gallery")
            run_id = run.info.run_id
    except Exception as e:  # noqa: BLE001 - the report is already written
        log.warning("MLflow logging skipped: %s", e)
    lineage.record("xai", lineage.history_path(cfg), dataset=cfg.data.name, model=cfg.model.name,
                   task=info["task"], metrics=metrics, report=str(path), mlflow_run_id=run_id)  # fmt: skip


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    explain(cfg)


if __name__ == "__main__":
    main()
