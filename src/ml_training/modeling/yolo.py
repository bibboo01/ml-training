"""YOLO object detection backend (Ultralytics) – same workflow as the tabular models:
Optuna search -> final training -> evaluation (mAP + TP/FP/FN per class) -> MLflow -> Excel.

    uv run mlt-train data=coco8 model=yolo11n
    uv run mlt-train data=helmet model=yolo11s hpo.n_trials=20 model.hpo_epochs=15
    uv run mlt-train data=helmet model=yolo11n hpo=none model.params.epochs=200
"""

from __future__ import annotations

import platform
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

from ml_training.detection_data import configure_ultralytics
from ml_training.hpo import param_importances, run_study, suggest_params, trials_dataframe
from ml_training.metrics import EvalResult
from ml_training.reporting.excel import write_report
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


def box_summary(metrics) -> dict[str, float]:
    b = metrics.box
    return {
        "map50": float(b.map50),
        "map50_95": float(b.map),
        "map75": float(b.map75),
        "precision": float(b.mp),
        "recall": float(b.mr),
        "f1": _f1(float(b.mp), float(b.mr)),
        "fitness": float(metrics.fitness),
    }


def build_eval(metrics_map, metrics_cm, class_names: list[str]) -> EvalResult:
    """Combine mAP metrics (low conf) with confusion-matrix counts (conf threshold)."""
    summary = box_summary(metrics_map)
    pc = detection_counts(metrics_cm.confusion_matrix.matrix, class_names)
    ap50 = dict.fromkeys(range(len(class_names)), np.nan)
    ap = dict.fromkeys(range(len(class_names)), np.nan)
    for j, c in enumerate(metrics_map.ap_class_index):
        ap50[int(c)] = float(metrics_map.box.ap50[j])
        ap[int(c)] = float(metrics_map.box.ap[j])
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


# --------------------------------------------------------------------------- mlflow model
class YoloPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: input = DataFrame with a ``path`` column (image files) -> detections."""

    def load_context(self, context):
        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        conf = (params or {}).get("conf", 0.25)
        rows = []
        for path in model_input["path"]:
            for r in self.model.predict(str(path), conf=conf, verbose=False):
                for box, c, s in zip(
                    r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), strict=True
                ):
                    rows.append(
                        {
                            "path": str(path),
                            "class": r.names[int(c)],
                            "conf": s,
                            "x1": box[0],
                            "y1": box[1],
                            "x2": box[2],
                            "y2": box[3],
                        }
                    )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- training
def resolve_weights(cfg: DictConfig) -> str:
    w = Path(str(cfg.model.weights))
    if w.is_absolute() and w.exists():
        return str(w)
    for cand in (Path(cfg.paths.root) / w, Path(cfg.paths.models) / "pretrained" / w.name):
        if cand.exists():
            return str(cand)
    target = Path(cfg.paths.models) / "pretrained" / w.name
    target.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    return str(attempt_download_asset(target))


def _train_kwargs(cfg: DictConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    kw = OmegaConf.to_container(cfg.model.params, resolve=True) or {}
    kw.update(overrides)
    return {k: v for k, v in kw.items() if v is not None}


def train_yolo(cfg: DictConfig) -> dict[str, float]:
    from ultralytics import YOLO

    configure_ultralytics(cfg)
    set_seed(cfg.seed)
    meta = read_json(Path(cfg.paths.processed) / "meta.json")
    data_yaml = meta["data_yaml"]
    class_names = meta["class_names"]
    weights = resolve_weights(cfg)
    hpo = cfg.hpo
    metric = cfg.model.get("hpo_metric") or hpo.metric
    ev = cfg.model.get("evaluation") or {}
    conf_thr = float(ev.get("conf", 0.25))
    runs_dir = ensure_dir(Path(cfg.paths.models) / cfg.data.name / cfg.model.name)
    out_dir = ensure_dir(cfg.paths.run_reports)
    eval_split = "test" if meta["n_rows"].get("test") else "val"

    experiment_id = setup_mlflow(cfg)
    run_name = f"{cfg.model.name}-{datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as parent:
        mlflow.set_tags(
            {
                "dataset": cfg.data.name,
                "model": cfg.model.name,
                "task": "detect",
                "backend": "ultralytics",
                "stage": "train",
                "host_os": platform.system(),
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config/config.yaml")
        mlflow.log_dict(meta, "config/data_meta.json")

        # ---------- 1. Optuna search (short runs) ----------
        study, best_params, val_summary = None, {}, None
        if hpo.enabled and hpo.n_trials > 0 and cfg.model.get("search_space"):

            def objective(trial: optuna.Trial) -> float:
                params = suggest_params(trial, cfg.model.search_space)
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
                        **_train_kwargs(cfg, {**params, "epochs": cfg.model.hpo_epochs}),
                    )
                    res = box_summary(model.trainer.validator.metrics)
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
            study = run_study(OmegaConf.merge(hpo, {"direction": "maximize"}), cfg.seed, objective)
            best_params = suggest_params(
                optuna.trial.FixedTrial(study.best_params), cfg.model.search_space
            )
            val_summary = study.best_trial.user_attrs.get("val_summary")
            mlflow.log_metric(f"best_val_{metric}", study.best_value)
            log.info("Best val %s = %.4f  params=%s", metric, study.best_value, best_params)

        # ---------- 2. Final training ----------
        model = YOLO(weights)
        t0 = time.perf_counter()
        train_kw = _train_kwargs(cfg, best_params)
        model.train(
            data=data_yaml,
            project=str(runs_dir),
            name="final",
            exist_ok=True,
            plots=True,
            **train_kw,
        )
        fit_seconds = time.perf_counter() - t0
        train_dir = Path(model.trainer.save_dir)
        best_pt = Path(model.trainer.best)
        log_params({f"model.{k}": v for k, v in train_kw.items()})
        mlflow.log_params(
            {
                "weights": Path(weights).name,
                "hpo.metric": metric,
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
        m_cm = best.val(
            # plots=True is required: Ultralytics only fills the confusion matrix when plotting
            name=f"eval_{eval_split}_conf{conf_thr}",
            plots=True,
            conf=conf_thr,
            **common,
        )
        result = build_eval(m_map, m_cm, class_names)
        result.summary["fit_seconds"] = fit_seconds
        mlflow.log_metrics({f"test_{k}": v for k, v in result.summary.items()})
        for row in result.per_class.itertuples():
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(row._1))
            mlflow.log_metrics(
                {
                    f"test_{safe}_{k}": getattr(row, k)
                    for k in ("TP", "FP", "FN", "precision", "recall", "f1")
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
        dataset_df = pd.DataFrame(
            [
                {
                    "class": n,
                    **{f"{s}_boxes": counts[s]["instances"].get(str(i), 0) for s in counts},
                }
                for i, n in enumerate(class_names)
            ]
        )
        dataset_df = dataset_df[dataset_df.iloc[:, 1:].sum(axis=1) > 0].reset_index(drop=True)
        run_info = {
            "task": "object detection (YOLO)",
            "dataset": cfg.data.name,
            "model": cfg.model.name,
            "weights": Path(weights).name,
            "classes": len(class_names),
            "images train/val/test": "/".join(
                str(meta["n_rows"].get(s, 0)) for s in ("train", "val", "test")
            ),
            "evaluated on": eval_split,
            "TP/FP/FN rule": f"conf >= {conf_thr}, IoU >= 0.45 (Ultralytics matching)",
            "hpo_metric": metric,
            "hpo_trials": len(study.trials) if study else 0,
            "epochs (final)": train_kw.get("epochs"),
            "mlflow_run_id": parent.info.run_id,
            "created": datetime.now().isoformat(timespec="seconds"),
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

        # ---------- 5. Log + register the model ----------
        mlflow.pyfunc.log_model(
            name="model",
            python_model=YoloPyfunc(),
            artifacts={"weights": str(model_path)},
            pip_requirements=["ultralytics", "pandas"],
            metadata={"class_names": class_names, "dataset": cfg.data.name, "task": "detect"},
            registered_model_name=(
                cfg.mlflow.registered_model_name if cfg.mlflow.register_model else None
            ),
        )

    s = result.summary
    log.info(
        "TEST(%s)  mAP50=%.4f  mAP50-95=%.4f  P=%.4f R=%.4f  TP=%d FP=%d FN=%d  -> run %s",
        eval_split,
        s["map50"],
        s["map50_95"],
        s["precision"],
        s["recall"],
        s["TP"],
        s["FP"],
        s["FN"],
        parent.info.run_id,
    )
    return s
