"""Tabular regression and time-series forecasting: tune (Optuna), train, evaluate, Excel, MLflow.

uv run mlt-prepare data=house-prices && uv run mlt-featurize data=house-prices
uv run mlt-train data=house-prices model=lightgbm
uv run mlt-train data=shop-sales model=xgboost              # forecast (lag features, time split)
uv run mlt-predict data=shop-sales model=xgboost +horizon=30  # next 30 days

Uses the same model configs as classification: ``regressor:`` is the class for numbers.
Optimises validation R² by default (hpo.metric=mae / rmse / mape ... minimises instead).
"""

from __future__ import annotations

import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf

from ml_training import lineage
from ml_training.features import load_splits
from ml_training.hpo import param_importances, run_study, suggest_params, trials_dataframe
from ml_training.metrics import LOWER_IS_BETTER, regression_metrics
from ml_training.modeling.factory import build_model, estimator_path, search_space
from ml_training.tracking import log_params, setup_mlflow
from ml_training.utils import ensure_dir, flatten_dict, log, set_seed, write_json

CLASSIFY_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "mcc"}


def hpo_metric(cfg: DictConfig) -> tuple[str, str]:
    m = str(cfg.hpo.get("metric") or "r2")
    if m in CLASSIFY_METRICS:  # default hpo config is for classification
        m = "r2"
    return m, "minimize" if m in LOWER_IS_BETTER else "maximize"


def permutation_importance_df(model, X: pd.DataFrame, y: np.ndarray, seed: int) -> pd.DataFrame:
    """How much the test error grows when one input column is shuffled (any model)."""
    from sklearn.inspection import permutation_importance

    Xs, ys = X, y
    if len(X) > 3000:  # X has a 0..n-1 index
        idx = X.sample(3000, random_state=seed).index
        Xs, ys = X.loc[idx], y[idx]
    r = permutation_importance(
        model, Xs, ys, n_repeats=5, random_state=seed, scoring="neg_mean_absolute_error"
    )
    df = pd.DataFrame(
        {"feature": Xs.columns, "mae_increase": r.importances_mean, "std": r.importances_std}
    )
    return df.sort_values("mae_increase", ascending=False).reset_index(drop=True)


def train_regression(cfg: DictConfig) -> dict[str, float]:
    from ml_training.reporting.excel import write_regression_report

    set_seed(cfg.seed)
    X_tr, y_tr, X_val, y_val, X_te, y_te, meta = load_splits(cfg.paths.processed)
    task, feats = meta["task"], meta["features"]
    y_tr, y_val, y_te = (np.asarray(v, float) for v in (y_tr, y_val, y_te))
    hpo = cfg.hpo
    metric, direction = hpo_metric(cfg)
    space = search_space(cfg.model, task)
    ts = meta.get("timeseries") or {}
    out_dir = ensure_dir(cfg.paths.run_reports)
    processed = Path(cfg.paths.processed)
    experiment_id = setup_mlflow(cfg)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{cfg.model.name}-{datetime.now():%Y%m%d-%H%M%S}",
        log_system_metrics=bool(cfg.mlflow.get("system_metrics", True)),
    ) as parent:
        mlflow.set_tags(
            {
                "dataset": cfg.data.name,
                "model": cfg.model.name,
                "backend": cfg.model.backend,
                "task": task,
                "stage": "train",
                "host_os": platform.system(),
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config/config.yaml")
        mlflow.log_dict(meta, "config/data_meta.json")
        split_files = sorted(processed.glob("*.csv"))
        lineage_info = lineage.log_context(
            [Path(cfg.paths.raw) / f"{cfg.data.name}.csv", *split_files],
            data_root=Path(cfg.paths.root),
            extra_artifacts=[f for f in split_files if f.stat().st_size < 50 * 2**20],
        )

        # ---------- 1. Optuna ----------
        study, best_params, val_summary = None, {}, None
        if hpo.enabled and hpo.n_trials > 0 and space:

            def objective(trial: optuna.Trial) -> float:
                params = suggest_params(trial, space)
                with mlflow.start_run(
                    experiment_id=experiment_id, run_name=f"trial-{trial.number}", nested=True
                ):
                    log_params(params)
                    model = build_model(cfg.model, feats, params, task)
                    t0 = time.perf_counter()
                    model.fit(X_tr, y_tr)
                    res = regression_metrics(y_val, model.predict(X_val))
                    res["fit_seconds"] = time.perf_counter() - t0
                    mlflow.log_metrics({f"val_{k}": v for k, v in res.items() if np.isfinite(v)})
                    trial.set_user_attr("val_summary", res)
                    return res[metric]

            log.info("Optuna: %d trials, %s val %s", hpo.n_trials, direction, metric)
            study = run_study(OmegaConf.merge(hpo, {"direction": direction}), cfg.seed, objective)
            best_params = suggest_params(optuna.trial.FixedTrial(study.best_params), space)
            val_summary = study.best_trial.user_attrs.get("val_summary")
            mlflow.log_metric(f"best_val_{metric}", study.best_value)
            log.info("Best val %s = %.4f  params=%s", metric, study.best_value, best_params)

        # ---------- 2. Final fit on train + val (forecast: val = the period before test) ----------
        X_fit, y_fit = (
            (pd.concat([X_tr, X_val]), np.concatenate([y_tr, y_val]))
            if hpo.refit_on_train_val
            else (X_tr, y_tr)
        )
        model = build_model(cfg.model, feats, best_params, task)
        t0 = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - t0
        all_params = {
            **(OmegaConf.to_container(cfg.model.params, resolve=True) or {}),
            **best_params,
        }
        log_params({f"model.{k}": v for k, v in all_params.items()})
        mlflow.log_params(
            {
                "hpo.n_trials": hpo.n_trials if hpo.enabled and space else 0,
                "hpo.metric": metric,
                "estimator": estimator_path(cfg.model, task),
                "seed": cfg.seed,
            }
        )

        # ---------- 3. Test ----------
        pred = model.predict(X_te)
        naive = X_te["lag_1"].to_numpy(float) if "lag_1" in X_te else None
        summary = regression_metrics(y_te, pred, naive)
        summary["fit_seconds"] = fit_seconds
        mlflow.log_metrics({f"test_{k}": v for k, v in summary.items() if np.isfinite(v)})

        # ---------- 4. Artifacts ----------
        model_path = Path(cfg.paths.run_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": model, "meta": meta, "params": all_params}, model_path)
        write_json(
            {"test": summary, "val_best": val_summary, "best_params": best_params},
            out_dir / "metrics.json",
        )

        preds = X_te.reset_index(drop=True).copy()
        preds.insert(0, "y_true", y_te)
        preds.insert(1, "y_pred", pred)
        preds.insert(2, "error", pred - y_te)
        preds.insert(3, "abs_error", np.abs(pred - y_te))
        preds.insert(
            4,
            "pct_error",
            np.where(
                np.abs(y_te) > 1e-12, (pred - y_te) / np.where(y_te == 0, 1, y_te) * 100, np.nan
            ),
        )
        if ts:  # time (+ series) first so the sheet reads like a forecast table
            first = [c for c in (ts.get("group_column"), ts["time_column"]) if c]
            preds = preds[first + [c for c in preds if c not in first]]
        preds.to_csv(out_dir / "test_predictions.csv", index=False)
        importance = permutation_importance_df(model, X_te.reset_index(drop=True), y_te, cfg.seed)
        importance.to_csv(out_dir / "feature_importance.csv", index=False)
        trials_df = trials_dataframe(study) if study else None
        importances = param_importances(study) if study and len(study.trials) > 1 else {}

        report_path = None
        if cfg.report.excel:
            run_info = {
                "dataset": cfg.data.name,
                "model": cfg.model.name,
                "task": task,
                "estimator": estimator_path(cfg.model, task),
                "target": meta["target"],
                "rows train/val/test": "{train}/{val}/{test}".format(**meta["n_rows"]),
                "hpo_metric": f"{metric} ({direction})",
                "hpo_trials": len(study.trials) if study else 0,
                "mlflow_run_id": parent.info.run_id,
                "created": datetime.now().isoformat(timespec="seconds"),
                **(
                    {
                        "time column": ts["time_column"],
                        "series column": ts.get("group_column") or "-",
                        "frequency": ts.get("freq"),
                        "lags": ts["lags"],
                        "rolling windows": ts["rolling"],
                    }
                    if ts
                    else {}
                ),
                **lineage_info,
            }
            report_path = write_regression_report(
                out_dir / cfg.report.filename,
                run_info,
                summary,
                best_params,
                preds,
                importance,
                trials_df,
                importances,
                flatten_dict(OmegaConf.to_container(cfg, resolve=True)),
                val_summary,
                ts,
            )
            log.info("Excel report: %s", report_path)
        for p in [
            out_dir / "metrics.json",
            out_dir / "test_predictions.csv",
            out_dir / "feature_importance.csv",
        ]:
            mlflow.log_artifact(str(p), artifact_path="reports")
        if report_path:
            mlflow.log_artifact(str(report_path), artifact_path="reports")
        if trials_df is not None:
            trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
            mlflow.log_artifact(str(out_dir / "optuna_trials.csv"), artifact_path="reports")

        # ---------- 5. Registry ----------
        sample = X_te.head(5)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=sample,
            serialization_format="cloudpickle",
            metadata={"task": task, "target": meta["target"], "dataset": cfg.data.name},
            registered_model_name=cfg.mlflow.registered_model_name
            if cfg.mlflow.register_model
            else None,
        )

    lineage.record(
        "train",
        history=lineage.history_path(cfg),
        dataset=cfg.data.name,
        model=cfg.model.name,
        task=task,
        hpo_trials=len(study.trials) if study else 0,
        metrics={k: v for k, v in summary.items() if isinstance(v, int | float) and np.isfinite(v)},
        model_file=str(model_path),
        report=str(report_path) if report_path else None,
        mlflow_run_id=parent.info.run_id,
        data_sha256=lineage_info.get("data sha256"),
    )
    log.info(
        "TEST %s  r2=%.4f  mae=%.4f  rmse=%.4f  mape=%.2f%%  -> run %s",
        task,
        summary["r2"],
        summary["mae"],
        summary["rmse"],
        summary["mape"],
        parent.info.run_id,
    )
    return summary


def predict_regression(
    cfg: DictConfig, pipeline, meta: dict, df: pd.DataFrame | None
) -> pd.DataFrame:
    """regress: one number per input row. forecast: input = history rows -> one-step predictions
    per row; without input -> the next `horizon` periods after the training data."""
    from ml_training import timeseries

    target = meta["target"]
    if meta["task"] == "forecast":
        ts = meta["timeseries"]
        steps = int(cfg.get("horizon") or cfg.data.get("horizon") or 14)
        if df is None:
            history = pd.read_csv(Path(cfg.paths.raw) / f"{cfg.data.name}.csv")
            log.info(
                "Forecasting %d periods (%s) after %s", steps, ts.get("freq"), ts.get("last_time")
            )
            return timeseries.forecast(pipeline, history, meta, steps)
        feats = timeseries.make_features(
            df, ts["time_column"], target, ts.get("group_column"), ts["lags"], ts["rolling"]
        )
        out = feats[[c for c in df.columns]].copy()
        out["prediction"] = pipeline.predict(feats)
        out.loc[feats[f"lag_{max(ts['lags'])}"].isna(), "prediction"] = (
            np.nan
        )  # not enough history yet
        return out
    out = df.copy()
    out["prediction"] = pipeline.predict(df.drop(columns=[target], errors="ignore"))
    if target in df:
        out["error"] = out["prediction"] - pd.to_numeric(df[target], errors="coerce")
    return out
