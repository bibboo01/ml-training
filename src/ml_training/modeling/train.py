"""Stage 3 – tune (Optuna), train, evaluate, track (MLflow) and report (Excel).

uv run mlt-train                                       # RF on breast_cancer, 30 trials
uv run mlt-train model=gradient_boosting hpo.n_trials=50
uv run mlt-train data=iris model=logistic_regression
uv run mlt-train hpo=none                              # no tuning, just model.params
uv run mlt-train -m model=random_forest,logistic_regression,svm   # compare models
uv run mlt-train data=coco8 model=yolo11n                          # YOLO object detection
"""

from __future__ import annotations

import platform
import time
from datetime import datetime
from pathlib import Path

import hydra
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf

from ml_training.features import load_splits
from ml_training.hpo import param_importances, run_study, suggest_params, trials_dataframe
from ml_training.metrics import evaluate, predict_with_threshold
from ml_training.modeling.factory import build_model
from ml_training.reporting.excel import write_report
from ml_training.reporting.plots import save_figures
from ml_training.tracking import log_params, setup_mlflow
from ml_training.utils import (
    CONFIG_DIR,
    ensure_dir,
    flatten_dict,
    log,
    set_seed,
    write_json,
)


def train(cfg: DictConfig) -> dict[str, float]:
    if cfg.model.backend == "ultralytics":  # image object detection (YOLO)
        from ml_training.modeling.yolo import train_yolo

        return train_yolo(cfg)

    set_seed(cfg.seed)
    X_tr, y_tr, X_val, y_val, X_te, y_te, meta = load_splits(cfg.paths.processed)
    class_names, n_classes = meta["class_names"], meta["n_classes"]
    feats = meta["features"]
    ev = cfg.evaluation
    hpo = cfg.hpo

    experiment_id = setup_mlflow(cfg)
    run_name = f"{cfg.model.name}-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir = ensure_dir(cfg.paths.run_reports)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as parent:
        mlflow.set_tags(
            {
                "dataset": cfg.data.name,
                "model": cfg.model.name,
                "backend": cfg.model.backend,
                "stage": "train",
                "host_os": platform.system(),
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config/config.yaml")
        mlflow.log_dict(meta, "config/data_meta.json")

        # ---------- 1. Hyperparameter search ----------
        study, best_params, val_summary = None, {}, None
        if hpo.enabled and hpo.n_trials > 0 and cfg.model.get("search_space"):

            def objective(trial: optuna.Trial) -> float:
                params = suggest_params(trial, cfg.model.search_space)
                with mlflow.start_run(
                    experiment_id=experiment_id, run_name=f"trial-{trial.number}", nested=True
                ):
                    mlflow.set_tag("optuna_trial", trial.number)
                    log_params(params)
                    model = build_model(cfg.model, feats, params)
                    t0 = time.perf_counter()
                    model.fit(X_tr, y_tr)
                    pred, proba = predict_with_threshold(model, X_val, n_classes, ev.threshold)
                    res = evaluate(y_val, pred, proba, class_names, ev.average)
                    res.summary["fit_seconds"] = time.perf_counter() - t0
                    mlflow.log_metrics({f"val_{k}": v for k, v in res.summary.items()})
                    trial.set_user_attr("val_summary", res.summary)
                    if hpo.metric not in res.summary:
                        raise KeyError(
                            f"hpo.metric={hpo.metric!r} not available: {list(res.summary)}"
                        )
                    return res.summary[hpo.metric]

            log.info("Optuna: %d trials, optimising val %s", hpo.n_trials, hpo.metric)
            study = run_study(hpo, cfg.seed, objective)
            best_params = suggest_params(
                optuna.trial.FixedTrial(study.best_params), cfg.model.search_space
            )
            val_summary = study.best_trial.user_attrs.get("val_summary")
            mlflow.log_metric(f"best_val_{hpo.metric}", study.best_value)
            log.info("Best val %s = %.4f  params=%s", hpo.metric, study.best_value, best_params)

        # ---------- 2. Final fit ----------
        if hpo.refit_on_train_val:
            X_fit, y_fit = pd.concat([X_tr, X_val]), np.concatenate([y_tr, y_val])
        else:
            X_fit, y_fit = X_tr, y_tr
        model = build_model(cfg.model, feats, best_params)
        t0 = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - t0
        all_params = {**OmegaConf.to_container(cfg.model.params, resolve=True), **best_params}
        log_params({f"model.{k}": v for k, v in all_params.items()})
        mlflow.log_params(
            {
                "hpo.n_trials": hpo.n_trials if hpo.enabled else 0,
                "hpo.metric": hpo.metric,
                "seed": cfg.seed,
            }
        )

        # ---------- 3. Test evaluation ----------
        pred, proba = predict_with_threshold(model, X_te, n_classes, ev.threshold)
        result = evaluate(y_te, pred, proba, class_names, ev.average)
        result.summary["fit_seconds"] = fit_seconds
        mlflow.log_metrics({f"test_{k}": v for k, v in result.summary.items()})
        for row in result.per_class.itertuples():
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(row._1))
            mlflow.log_metrics(
                {
                    f"test_{safe}_{m}": getattr(row, m)
                    for m in ("TP", "FP", "FN", "TN", "precision", "recall", "f1")
                }
            )

        # ---------- 4. Artifacts: model, figures, Excel ----------
        model_path = Path(cfg.paths.run_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": model, "meta": meta, "params": all_params}, model_path)

        write_json(
            {"test": result.summary, "val_best": val_summary, "best_params": best_params},
            out_dir / "metrics.json",
        )
        result.per_class.to_csv(out_dir / "per_class.csv", index=False)
        figs = save_figures(result, out_dir / "figures")

        trials_df = trials_dataframe(study) if study else None
        importances = param_importances(study) if study and len(study.trials) > 1 else {}

        report_path = None
        if cfg.report.excel:
            preds = X_te.reset_index(drop=True).copy()
            preds.insert(0, "y_true", [class_names[i] for i in y_te])
            preds.insert(1, "y_pred", [class_names[i] for i in pred])
            preds.insert(2, "correct", preds.y_true == preds.y_pred)
            if proba is not None:
                for i, c in enumerate(class_names):
                    preds.insert(3 + i, f"proba_{c}", proba[:, i])
            run_info = {
                "dataset": cfg.data.name,
                "model": cfg.model.name,
                "class_path": cfg.model.class_path,
                "classes": ", ".join(class_names),
                "rows train/val/test": "{train}/{val}/{test}".format(**meta["n_rows"]),
                "hpo_metric": f"{hpo.metric} ({ev.average})",
                "hpo_trials": len(study.trials) if study else 0,
                "mlflow_run_id": parent.info.run_id,
                "created": datetime.now().isoformat(timespec="seconds"),
            }
            cfg_flat = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            report_path = write_report(
                out_dir / cfg.report.filename,
                run_info,
                result,
                best_params,
                trials_df,
                importances,
                preds,
                cfg_flat,
                val_summary,
            )
            log.info("Excel report: %s", report_path)

        for p in [out_dir / "metrics.json", out_dir / "per_class.csv", *figs]:
            mlflow.log_artifact(str(p), artifact_path="reports")
        if report_path:
            mlflow.log_artifact(str(report_path), artifact_path="reports")
        if trials_df is not None:
            trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
            mlflow.log_artifact(str(out_dir / "optuna_trials.csv"), artifact_path="reports")

        # ---------- 5. Log + register the model ----------
        sample = X_te.head(5)
        signature = infer_signature(sample, model.predict(sample))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=sample,
            # cloudpickle supports any estimator, incl. custom ones like LightningMLPClassifier
            serialization_format="cloudpickle",
            metadata={"class_names": class_names, "dataset": cfg.data.name},
            registered_model_name=(
                cfg.mlflow.registered_model_name if cfg.mlflow.register_model else None
            ),
        )

    s = result.summary
    log.info(
        "TEST  acc=%.4f  f1=%.4f  TP=%d FP=%d FN=%d TN=%d  -> run %s",
        s["accuracy"],
        s["f1"],
        s["TP"],
        s["FP"],
        s["FN"],
        s["TN"],
        parent.info.run_id,
    )
    return s


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> float:
    summary = train(cfg)
    # Returning a float lets Hydra sweepers (e.g. hydra-optuna-sweeper) optimise across runs.
    return float(summary.get(cfg.hpo.metric, summary["f1"]))


if __name__ == "__main__":
    main()
