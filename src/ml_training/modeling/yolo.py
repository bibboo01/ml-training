"""Ultralytics YOLO backend for every image task – detect, segment, classify, pose, obb.
Same workflow as the tabular models:
Optuna search -> final training -> evaluation (mAP / top-1 + TP/FP/FN per class) -> MLflow -> Excel.

The task comes from the dataset (data/processed/<name>/meta.json); stock weights are switched
automatically, e.g. model=yolo11n on a segmentation dataset trains yolo11n-seg.pt.

    uv run mlt-train data=coco8 model=yolo11n
    uv run mlt-train data=coco8-seg model=yolo11s        # segmentation
    uv run mlt-train data=mnist160 model=yolo11n         # classification
    uv run mlt-train data=helmet model=yolo11s hpo.n_trials=20 model.hpo_epochs=15
    uv run mlt-train data=helmet model=yolo11n hpo=none model.params.epochs=200
    uv run python -m ml_training.resume run helmet yolo11s   # continue a stopped training
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ml_training import lineage
from ml_training.detection_data import configure_ultralytics
from ml_training.hpo import param_importances, run_study, suggest_params, trials_dataframe
from ml_training.metrics import EvalResult
from ml_training.reporting.excel import write_report
from ml_training.runstate import STATE_FILE, hydra_overrides, write_state
from ml_training.tracking import log_params, setup_mlflow
from ml_training.utils import ensure_dir, flatten_dict, log, read_json, set_seed, write_json


# --------------------------------------------------------------------------- metrics
def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def detection_counts(matrix: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Per-class TP/FP/FN from an Ultralytics detection confusion matrix.

    ``matrix`` is indexed [predicted, true] with an extra last row/column for *background*:
    FP = predicted as the class but wrong (other class or background),
    FN = ground-truth box of the class that was missed or given another class.
    TN is not defined for object detection (infinitely many empty boxes).
    """
    nc = len(class_names)
    rows = []
    for i, name in enumerate(class_names):
        tp = int(matrix[i, i])
        fp = int(matrix[i, :].sum() - tp)
        fn = int(matrix[:, i].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "class": name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "support": tp + fn,
                "precision": p,
                "recall": r,
                "f1": _f1(p, r),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["background_fp"] = int(matrix[:nc, nc].sum())
    df.attrs["background_fn"] = int(matrix[nc, :nc].sum())
    return df


TASK_SUFFIX = {"detect": "", "segment": "-seg", "classify": "-cls", "pose": "-pose", "obb": "-obb"}
TASK_LABEL = {
    "detect": "object detection (YOLO)",
    "segment": "instance segmentation (YOLO)",
    "classify": "image classification (YOLO)",
    "pose": "pose / keypoints (YOLO)",
    "obb": "oriented bounding boxes (YOLO)",
}
# which Ultralytics metric family is the main one per task: B = box, M = mask, P = pose
MAIN_FAMILY = {"detect": "B", "obb": "B", "segment": "M", "pose": "P"}
FAMILY = {"B": ("box", "box"), "M": ("mask", "seg"), "P": ("pose", "pose")}  # prefix, attribute
CLASSIFY_METRICS = ("top1", "top5", "accuracy", "fitness")


def summarize(metrics, task: str = "detect") -> dict[str, float]:
    """Flat summary of an Ultralytics metrics object; map50/map50_95/precision/recall are the
    task's main family (mask for segment, keypoints for pose) and box_* are added alongside."""
    if task == "classify":
        top1, top5 = float(metrics.top1), float(metrics.top5)
        return {"top1": top1, "top5": top5, "accuracy": top1, "fitness": float(metrics.fitness)}
    main = getattr(metrics, FAMILY[MAIN_FAMILY[task]][1])
    p, r = float(main.mp), float(main.mr)
    out = {
        "map50": float(main.map50),
        "map50_95": float(main.map),
        "map75": float(main.map75),
        "precision": p,
        "recall": r,
        "f1": _f1(p, r),
    }
    if MAIN_FAMILY[task] != "B":
        out.update({"box_map50": float(metrics.box.map50), "box_map50_95": float(metrics.box.map)})
    out["fitness"] = float(metrics.fitness)
    return out


box_summary = summarize  # backwards-compatible name


def classify_counts(matrix: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """One-vs-rest TP/FP/FN/TN per class from a classification confusion matrix [pred, true]."""
    m = np.asarray(matrix)[: len(class_names), : len(class_names)]
    total = int(m.sum())
    rows = []
    for i, name in enumerate(class_names):
        tp = int(m[i, i])
        fp = int(m[i, :].sum() - tp)
        fn = int(m[:, i].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "class": name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": total - tp - fp - fn,
                "support": tp + fn,
                "precision": p,
                "recall": r,
                "f1": _f1(p, r),
            }
        )
    return pd.DataFrame(rows)


def build_classify_eval(metrics, class_names: list[str]) -> EvalResult:
    summary = summarize(metrics, "classify")
    m = metrics.confusion_matrix.matrix
    pc = classify_counts(m, class_names)
    summary.update(
        {
            "precision": float(pc.precision.mean()),
            "recall": float(pc.recall.mean()),
            "f1": float(pc.f1.mean()),
        }
    )
    summary.update({k: int(pc[k].sum()) for k in ("TP", "FP", "FN", "TN")})
    n = len(class_names)
    confusion = pd.DataFrame(
        np.asarray(m)[:n, :n].T.astype(int),  # rows = true, cols = predicted
        index=[f"true: {c}" for c in class_names],
        columns=[f"pred: {c}" for c in class_names],
    )
    return EvalResult(summary=summary, per_class=pc, confusion=confusion)


def build_eval(metrics_map, metrics_cm, class_names: list[str], task: str = "detect") -> EvalResult:
    """Combine mAP metrics (low conf) with confusion-matrix counts (conf threshold)."""
    if task == "classify":
        return build_classify_eval(metrics_map, class_names)
    summary = summarize(metrics_map, task)
    pc = detection_counts(metrics_cm.confusion_matrix.matrix, class_names)
    main = getattr(metrics_map, FAMILY[MAIN_FAMILY[task]][1])
    ap50 = dict.fromkeys(range(len(class_names)), np.nan)
    ap = dict.fromkeys(range(len(class_names)), np.nan)
    for j, c in enumerate(metrics_map.ap_class_index):
        ap50[int(c)] = float(main.ap50[j])
        ap[int(c)] = float(main.ap[j])
    pc["AP50"] = [ap50[i] for i in range(len(class_names))]
    pc["AP50_95"] = [ap[i] for i in range(len(class_names))]
    # Large label sets (e.g. COCO) – keep classes that appear in GT or predictions.
    pc = pc[(pc.support > 0) | (pc.FP > 0)].reset_index(drop=True)
    summary.update({k: int(pc[k].sum()) for k in ("TP", "FP", "FN")})

    m = metrics_cm.confusion_matrix.matrix
    keep = [class_names.index(c) for c in pc["class"]]
    idx = keep + [len(class_names)]
    labels = [class_names[i] for i in keep] + ["background"]
    confusion = pd.DataFrame(
        m[np.ix_(idx, idx)].T.astype(int),  # transpose -> rows = true, cols = predicted
        index=[f"true: {c}" for c in labels],
        columns=[f"pred: {c}" for c in labels],
    )
    return EvalResult(summary=summary, per_class=pc, confusion=confusion)


def result_rows(r) -> list[dict[str, Any]]:
    """One row per prediction of an Ultralytics ``Results`` object, for any task."""
    if r.probs is not None:  # classify: top-1 (+ top-5 as text)
        top5 = ", ".join(
            f"{r.names[int(i)]}:{c:.3f}"
            for i, c in zip(r.probs.top5, r.probs.top5conf.tolist(), strict=True)
        )
        return [
            {
                "source": r.path,
                "class": r.names[int(r.probs.top1)],
                "conf": float(r.probs.top1conf),
                "top5": top5,
            }
        ]
    rows: list[dict[str, Any]] = []
    if r.obb is not None:  # rotated boxes: centre, size, angle (radians)
        for (cx, cy, w, h, a), c, s in zip(
            r.obb.xywhr.tolist(), r.obb.cls.tolist(), r.obb.conf.tolist(), strict=True
        ):
            rows.append(
                {"source": r.path, "class": r.names[int(c)], "conf": s}
                | {"cx": cx, "cy": cy, "w": w, "h": h, "angle": a}
            )
        return rows
    if r.boxes is None:
        return rows
    polys = r.masks.xy if r.masks is not None else None
    kpts = r.keypoints.data.tolist() if r.keypoints is not None else None
    for k, (box, c, s) in enumerate(
        zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), strict=True)
    ):
        row = {"source": r.path, "class": r.names[int(c)], "conf": s}
        row |= {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
        if polys is not None:
            row["polygon"] = json.dumps(np.round(polys[k], 1).tolist())
        if kpts is not None:
            row["keypoints"] = json.dumps([[round(v, 2) for v in p] for p in kpts[k]])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- mlflow model
class YoloPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: input = DataFrame with a ``path`` column (image files) -> predictions."""

    def load_context(self, context):
        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        conf = (params or {}).get("conf", 0.25)
        rows = []
        for path in model_input["path"]:
            for r in self.model.predict(str(path), conf=conf, verbose=False):
                rows += result_rows(r)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- training
def task_weights(name: str, task: str) -> str:
    """Stock detection weights -> the task's variant: yolo11n.pt + segment -> yolo11n-seg.pt."""
    p = Path(name)
    if TASK_SUFFIX.get(task) and re.fullmatch(r"yolo(v?\d+)[nsmlx]", p.stem):
        return str(p.with_name(p.stem + TASK_SUFFIX[task] + p.suffix))
    return name


def resolve_weights(cfg: DictConfig, task: str = "detect") -> str:
    w = Path(task_weights(str(cfg.model.weights), task))
    if w.is_absolute() and w.exists():
        return str(w)
    for cand in (Path(cfg.paths.root) / w, Path(cfg.paths.models) / "pretrained" / w.name):
        if cand.exists():
            return str(cand)
    target = Path(cfg.paths.models) / "pretrained" / w.name
    target.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    return str(attempt_download_asset(target))


def _train_kwargs(
    cfg: DictConfig, overrides: dict[str, Any], task: str = "detect"
) -> dict[str, Any]:
    kw = OmegaConf.to_container(cfg.model.params, resolve=True) or {}
    per_task = cfg.model.get("task_params") or {}
    if per_task.get(task):  # e.g. classify -> imgsz 224
        kw.update(OmegaConf.to_container(per_task[task], resolve=True))
    kw.update(overrides)
    return {k: v for k, v in kw.items() if v is not None}


def unfinished_checkpoint(train_dir: Path) -> Path | None:
    """last.pt of a training that stopped before the end (Ultralytics strips the optimizer from
    last.pt when a training finishes, so an optimizer still inside = can be resumed)."""
    last = Path(train_dir) / "weights" / "last.pt"
    if not last.exists():
        return None
    import torch

    try:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - e.g. file half-written when the PC lost power
        log.warning("Cannot read %s (%s) – the final training will start again", last, e)
        return None
    ok = isinstance(ckpt, dict) and ckpt.get("optimizer") is not None
    return last if ok and ckpt.get("epoch", -1) >= 0 else None


def train_yolo(cfg: DictConfig) -> dict[str, float]:
    from ultralytics import YOLO

    configure_ultralytics(cfg)
    set_seed(cfg.seed)
    meta = read_json(Path(cfg.paths.processed) / "meta.json")
    data_yaml = meta["data_yaml"]
    class_names = meta["class_names"]
    task = meta.get("task", "detect")
    weights = resolve_weights(cfg, task)
    tune = cfg.get("tune") or {}
    tune_name = str(tune.get("name") or "default")
    space = tune.get("search_space") or cfg.model.get("search_space")  # preset replaces model's
    hpo = cfg.hpo
    metric = cfg.model.get("hpo_metric") or hpo.metric
    if task == "classify" and metric not in CLASSIFY_METRICS:
        metric = "top1"
    elif task != "classify" and metric in ("top1", "top5", "accuracy"):
        metric = "map50_95"
    ev = cfg.model.get("evaluation") or {}
    conf_thr = float(ev.get("conf", 0.25))
    runs_dir = ensure_dir(Path(cfg.paths.models) / cfg.data.name / cfg.model.name)
    out_dir = ensure_dir(cfg.paths.run_reports)
    eval_split = "test" if meta["n_rows"].get("test") else "val"

    # resume=true: finished Optuna trials are kept and an interrupted final training continues
    # from its last epoch (models/<dataset>/<model>/final/weights/last.pt)
    resume = bool(cfg.get("resume"))
    resume_ckpt = unfinished_checkpoint(runs_dir / "final") if resume else None
    skip_fit = False  # stopped during evaluation: the trained best.pt is reused
    if resume:
        prev = read_json(runs_dir / STATE_FILE) if (runs_dir / STATE_FILE).exists() else {}
        best_file = runs_dir / "final" / "weights" / "best.pt"
        skip_fit = prev.get("stage") == "evaluate" and best_file.exists()
        write_state(runs_dir, status="running", resumes=int(prev.get("resumes", 0)) + 1)
        how = "keeping finished Optuna trials, final training starts fresh"
        if resume_ckpt:
            how = f"final training from {resume_ckpt}"
        elif skip_fit:
            how = f"training already finished, evaluating {best_file}"
        log.info("Resume: %s", how)
    else:
        write_state(runs_dir, status="running", stage="start", dataset=cfg.data.name,
                    model=cfg.model.name, task=task, tune=tune_name, overrides=hydra_overrides(),
                    started=datetime.now().isoformat(timespec="seconds"), resumes=0)  # fmt: skip

    experiment_id = setup_mlflow(cfg)
    run_name = f"{cfg.model.name}-{datetime.now():%Y%m%d-%H%M%S}"
    processed = Path(cfg.paths.processed)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        log_system_metrics=bool(cfg.mlflow.get("system_metrics", True)),
    ) as parent:
        mlflow.set_tags(
            {
                "dataset": cfg.data.name,
                "model": cfg.model.name,
                "task": task,
                "backend": "ultralytics",
                "stage": "train",
                "host_os": platform.system(),
                "resumed": str(resume),
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config/config.yaml")
        mlflow.log_dict(meta, "config/data_meta.json")
        # lineage: git, environment, data fingerprint + the exact split lists used
        split_files = [processed / n for n in ("data.yaml", "train.txt", "val.txt", "test.txt")]
        lineage_info = lineage.log_context(
            lineage.yolo_data_files(processed, task), extra_artifacts=split_files
        )

        # ---------- 1. Optuna search (short runs) ----------
        study, best_params, val_summary = None, {}, None
        if hpo.enabled and hpo.n_trials > 0 and space:  # on resume only the missing trials run
            write_state(runs_dir, stage="hpo")

            def objective(trial: optuna.Trial) -> float:
                params = suggest_params(trial, space)
                with mlflow.start_run(
                    experiment_id=experiment_id, run_name=f"trial-{trial.number}", nested=True
                ):
                    mlflow.set_tag("optuna_trial", trial.number)
                    log_params(params)
                    model = YOLO(weights)
                    t0 = time.perf_counter()
                    model.train(
                        data=data_yaml,
                        project=str(runs_dir / "hpo"),
                        name=f"trial_{trial.number}",
                        exist_ok=True,
                        plots=False,
                        verbose=False,
                        **_train_kwargs(cfg, {**params, "epochs": cfg.model.hpo_epochs}, task),
                    )
                    res = summarize(model.trainer.validator.metrics, task)
                    res["fit_seconds"] = time.perf_counter() - t0
                    mlflow.log_metrics({f"val_{k}": v for k, v in res.items()})
                    trial.set_user_attr("val_summary", res)
                    return res[metric]

            log.info(
                "Optuna: %d trials x %d epochs, optimising val %s",
                hpo.n_trials,
                cfg.model.hpo_epochs,
                metric,
            )
            study = run_study(
                OmegaConf.merge(hpo, {"direction": "maximize"}), cfg.seed, objective, resume=resume
            )
            best_params = suggest_params(optuna.trial.FixedTrial(study.best_params), space)
            val_summary = study.best_trial.user_attrs.get("val_summary")
            mlflow.log_metric(f"best_val_{metric}", study.best_value)
            log.info("Best val %s = %.4f  params=%s", metric, study.best_value, best_params)

        # ---------- 2. Final training ----------
        write_state(runs_dir, stage="final")
        t0 = time.perf_counter()
        if skip_fit:
            train_dir = runs_dir / "final"
            best_pt = train_dir / "weights" / "best.pt"
            done = YOLO(str(best_pt)).ckpt.get("train_args") or {}
            train_kw = {k: done[k] for k in _train_kwargs(cfg, best_params, task) if k in done}
        elif resume_ckpt is not None:
            model = YOLO(str(resume_ckpt))
            model.train(resume=True, data=data_yaml)  # same settings as before, from last epoch
            args = model.trainer.args
            keys = _train_kwargs(cfg, best_params, task)
            train_kw = {k: getattr(args, k) for k in keys if hasattr(args, k)}
            mlflow.set_tag("resumed_from", str(resume_ckpt))
        else:
            model = YOLO(weights)
            train_kw = _train_kwargs(cfg, best_params, task)
            model.train(
                data=data_yaml,
                project=str(runs_dir),
                name="final",
                exist_ok=True,
                plots=True,
                **train_kw,
            )
        fit_seconds = time.perf_counter() - t0
        if not skip_fit:
            train_dir = Path(model.trainer.save_dir)
            best_pt = Path(model.trainer.best)
        write_state(runs_dir, stage="evaluate")
        log_params({f"model.{k}": v for k, v in train_kw.items()})
        mlflow.log_params(
            {
                "weights": Path(weights).name,
                "hpo.metric": metric,
                "tune": tune_name,
                "hpo.n_trials": hpo.n_trials if hpo.enabled else 0,
            }
        )

        # ---------- 3. Evaluation on test (or val) ----------
        best = YOLO(str(best_pt))
        common = {
            "data": data_yaml,
            "split": eval_split,
            "project": str(runs_dir),
            "exist_ok": True,
            "verbose": False,
        }
        for k in ("imgsz", "batch", "device", "workers"):
            if train_kw.get(k) is not None:
                common[k] = train_kw[k]
        m_map = best.val(name=f"eval_{eval_split}", plots=True, **common)
        if task == "classify":  # no confidence threshold: top-1 vs true class
            m_cm = m_map
        else:
            m_cm = best.val(
                # plots=True is required: Ultralytics only fills the confusion matrix when plotting
                name=f"eval_{eval_split}_conf{conf_thr}",
                plots=True,
                conf=conf_thr,
                **common,
            )
        result = build_eval(m_map, m_cm, class_names, task)
        result.summary["fit_seconds"] = fit_seconds
        mlflow.log_metrics({f"test_{k}": v for k, v in result.summary.items()})
        for row in result.per_class.itertuples():
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(row._1))
            mlflow.log_metrics(
                {
                    f"test_{safe}_{k}": getattr(row, k)
                    for k in ("TP", "FP", "FN", "precision", "recall", "f1")
                    if hasattr(row, k)
                }
            )

        # ---------- 4. Files: weights, metrics, Excel ----------
        model_path = Path(cfg.paths.run_model).with_suffix(".pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, model_path)
        write_json(
            {
                "test": result.summary,
                "val_best": val_summary,
                "best_params": best_params,
                "eval_split": eval_split,
            },
            out_dir / "metrics.json",
        )
        result.per_class.to_csv(out_dir / "per_class.csv", index=False)
        fig_dir = ensure_dir(out_dir / "figures")
        # PR/F1 curves from the mAP run; confusion matrices from the conf-threshold run
        curves = [p for p in sorted(Path(m_map.save_dir).glob("*curve*.png"))]
        cms = sorted(Path(m_cm.save_dir).glob("confusion_matrix*.png"))
        figs = []
        for src in [*cms, *curves, train_dir / "results.png"]:
            if src.exists():
                dst = fig_dir / src.name
                shutil.copy2(src, dst)
                figs.append(dst)

        history = None
        if (train_dir / "results.csv").exists():
            history = pd.read_csv(train_dir / "results.csv")
            history.columns = [c.strip() for c in history.columns]
        trials_df = trials_dataframe(study) if study else None
        importances = param_importances(study) if study and len(study.trials) > 1 else {}
        counts = meta["counts"]
        unit = "images" if task == "classify" else "objects"
        dataset_df = pd.DataFrame(
            [
                {
                    "class": n,
                    **{f"{s}_{unit}": counts[s]["instances"].get(str(i), 0) for s in counts},
                }
                for i, n in enumerate(class_names)
            ]
        )
        dataset_df = dataset_df[dataset_df.iloc[:, 1:].sum(axis=1) > 0].reset_index(drop=True)
        run_info = {
            "task": TASK_LABEL.get(task, task),
            "dataset": cfg.data.name,
            "model": cfg.model.name,
            "weights": Path(weights).name,
            "classes": len(class_names),
            "images train/val/test": "/".join(
                str(meta["n_rows"].get(s, 0)) for s in ("train", "val", "test")
            ),
            "evaluated on": eval_split,
            "main metric": {
                "segment": "map50 / map50_95 = mask mAP (box mAP in box_map50*)",
                "pose": "map50 / map50_95 = keypoint mAP (box mAP in box_map50*)",
                "obb": "map50 / map50_95 = rotated-box mAP",
                "classify": "top1 / top5 accuracy",
            }.get(task, "map50 / map50_95 = box mAP"),
            "TP/FP/FN rule": (
                "top-1 class vs true class, one-vs-rest per class"
                if task == "classify"
                else f"conf >= {conf_thr}, box IoU >= 0.45 (Ultralytics matching)"
            ),
            "tuning preset": tune_name,
            "hpo_metric": metric,
            "hpo_trials": len(study.trials) if study else 0,
            "epochs (final)": train_kw.get("epochs"),
            "mlflow_run_id": parent.info.run_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            **lineage_info,
        }
        report_path = None
        if cfg.report.excel:
            report_path = write_report(
                out_dir / cfg.report.filename,
                run_info,
                result,
                best_params,
                trials_df,
                importances,
                None,
                flatten_dict(OmegaConf.to_container(cfg, resolve=True)),
                val_summary,
                history=history,
                images=figs,
                dataset=dataset_df,
            )
            log.info("Excel report: %s", report_path)

        for p in [out_dir / "metrics.json", out_dir / "per_class.csv", *figs]:
            mlflow.log_artifact(str(p), artifact_path="reports")
        if report_path:
            mlflow.log_artifact(str(report_path), artifact_path="reports")
        if trials_df is not None:
            trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
            mlflow.log_artifact(str(out_dir / "optuna_trials.csv"), artifact_path="reports")
        if (train_dir / "results.csv").exists():
            mlflow.log_artifact(str(train_dir / "results.csv"), artifact_path="training")
        if (train_dir / "args.yaml").exists():
            mlflow.log_artifact(str(train_dir / "args.yaml"), artifact_path="training")
        lineage.log_epochs(train_dir / "results.csv")  # mAP / loss / lr curves per epoch
        if cfg.mlflow.get("log_weights", True):
            # whole Ultralytics output: plots, label stats, val batches, last.pt
            for run_dir in [train_dir, Path(m_map.save_dir), Path(m_cm.save_dir)]:
                for f in sorted(run_dir.glob("*")):
                    if f.is_file():
                        mlflow.log_artifact(str(f), artifact_path=f"ultralytics/{run_dir.name}")
            last_pt = train_dir / "weights" / "last.pt"
            if last_pt.exists():
                mlflow.log_artifact(str(last_pt), artifact_path="weights")

        # ---------- 5. Log + register the model ----------
        mlflow.pyfunc.log_model(
            name="model",
            python_model=YoloPyfunc(),
            artifacts={"weights": str(model_path)},
            pip_requirements=["ultralytics", "pandas"],
            metadata={"class_names": class_names, "dataset": cfg.data.name, "task": task},
            registered_model_name=(
                cfg.mlflow.registered_model_name if cfg.mlflow.register_model else None
            ),
        )

    s = result.summary
    lineage.record(
        "train",
        history=lineage.history_path(cfg),
        dataset=cfg.data.name,
        model=cfg.model.name,
        task=task,
        tune=tune_name,
        weights=Path(weights).name,
        epochs=train_kw.get("epochs"),
        hpo_trials=len(study.trials) if study else 0,
        metrics={k: v for k, v in s.items() if isinstance(v, int | float)},
        model_file=str(model_path),
        report=str(report_path) if report_path else None,
        mlflow_run_id=parent.info.run_id,
        data_sha256=lineage_info.get("data sha256"),
        resumed=resume,
    )
    write_state(runs_dir, status="finished", stage="done", mlflow_run_id=parent.info.run_id)
    keys = ["top1", "top5"] if task == "classify" else ["map50", "map50_95", "precision", "recall"]
    keys += [k for k in ("TP", "FP", "FN", "TN") if k in s]
    log.info(
        "TEST(%s) %s  %s  -> run %s",
        eval_split,
        task,
        "  ".join(f"{k}={s[k]:.4f}" if isinstance(s[k], float) else f"{k}={s[k]}" for k in keys),
        parent.info.run_id,
    )
    return s
