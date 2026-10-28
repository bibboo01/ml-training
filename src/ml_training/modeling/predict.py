"""Batch prediction with a trained model.

uv run mlt-predict +input=data/external/new_rows.csv
uv run mlt-predict data=shop-sales model=lightgbm +horizon=30   # forecast the next 30 periods
uv run mlt-predict +input=new.csv +output=preds.csv \
    +model_uri=models:/breast_cancer-random_forest/latest

YOLO, any task (input = image, folder of images, video or URL):
uv run mlt-predict data=coco8 model=yolo11n +input=data/raw/coco8/images/val
uv run mlt-predict data=helmet model=yolo11n +input=D:/cam/clip.mp4 +conf=0.4
"""

from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig

from ml_training import lineage
from ml_training.utils import CONFIG_DIR, log, sqlite_uri


def track_prediction(cfg: DictConfig, info: dict, out_csv: Path, df: pd.DataFrame) -> None:
    """History of every prediction: an MLflow run (stage=predict) + a logs/history.jsonl line."""
    counts = df["class"].value_counts().to_dict() if "class" in df else {}
    if "prediction" in df and not pd.api.types.is_float_dtype(df["prediction"]):
        counts = df["prediction"].value_counts().to_dict()
    run_id = None
    if cfg.mlflow.get("log_predictions", True):
        import mlflow

        from ml_training.tracking import setup_mlflow

        exp = setup_mlflow(cfg)
        with mlflow.start_run(
            experiment_id=exp,
            run_name=f"predict-{cfg.model.name}-{pd.Timestamp.now():%Y%m%d-%H%M%S}",
        ) as run:
            mlflow.set_tags({"stage": "predict", "dataset": cfg.data.name, "model": cfg.model.name})
            mlflow.log_params({k: str(v)[:500] for k, v in info.items()})
            mlflow.log_metric("n_predictions", len(df))
            for c, n in counts.items():
                safe = "".join(ch if ch.isalnum() else "_" for ch in str(c))
                mlflow.log_metric(f"count_{safe}", int(n))
            lineage.log_context()
            mlflow.log_artifact(str(out_csv), artifact_path="predictions")
            run_id = run.info.run_id
    lineage.record(
        "predict",
        lineage.history_path(cfg),
        dataset=cfg.data.name,
        model=cfg.model.name,
        **info,
        n_predictions=len(df),
        counts={str(k): int(v) for k, v in counts.items()},
        output=str(out_csv),
        mlflow_run_id=run_id,
    )


def predict_yolo(cfg: DictConfig) -> Path:
    """Any YOLO task; saves annotated images/videos and a CSV with one row per prediction
    (boxes; + polygon for segment, + keypoints for pose; rotated boxes for obb; top-5 for
    classify)."""
    from ultralytics import YOLO

    from ml_training.detection_data import configure_ultralytics

    configure_ultralytics(cfg)
    root = Path(cfg.paths.root)
    weights = Path(cfg.get("weights") or Path(cfg.paths.run_model).with_suffix(".pt"))
    source = Path(str(cfg.input))
    source = str(source if source.is_absolute() or "://" in str(cfg.input) else root / source)
    out_dir = Path(cfg.paths.run_reports) / "predict"
    model = YOLO(str(weights))
    from ml_training.modeling.yolo import result_rows

    rows = []
    results = model.predict(
        source=source,
        conf=float(cfg.get("conf", 0.25)),
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        stream=True,
        verbose=False,
    )
    for r in results:
        rows += result_rows(r)
    out_path = Path(cfg.get("output") or out_dir / "detections.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source", "class", "conf"])
    df.to_csv(out_path, index=False)
    log.info("%d predictions -> %s  (annotated files in %s)", len(rows), out_path, out_dir)
    info = {
        "input": source,
        "conf": float(cfg.get("conf", 0.25)),
        "weights": str(weights),
        "weights_sha256": lineage.fingerprint([weights])["sha256"],
        "annotated_dir": str(out_dir),
    }
    track_prediction(cfg, info, out_path, df)
    return out_path


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    from ml_training.backends import model_backend

    backend = model_backend(cfg)  # YOLO and other non-tabular model families
    if backend:
        backend.predict(cfg)
        return
    root = Path(cfg.paths.root)
    df, inp = None, None
    if cfg.get("input"):
        inp = Path(cfg.input)
        inp = inp if inp.is_absolute() else root / inp
        df = pd.read_excel(inp) if inp.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(inp)

    model_uri = cfg.get("model_uri")
    if model_uri:  # load from the MLflow Model Registry
        import mlflow

        mlflow.set_tracking_uri(sqlite_uri(cfg.mlflow.tracking_uri))
        pipeline = mlflow.sklearn.load_model(model_uri)
        metadata = mlflow.models.get_model_info(model_uri).metadata or {}
        class_names = metadata.get("class_names") or [str(c) for c in pipeline.classes_]
    else:  # load the local joblib saved by mlt-train
        bundle = joblib.load(cfg.paths.run_model)
        pipeline, class_names = bundle["pipeline"], bundle["meta"]["class_names"]

    meta = {} if model_uri else bundle["meta"]
    if meta.get("task") in ("regress", "forecast"):
        from ml_training.modeling.regression import predict_regression

        out = predict_regression(cfg, pipeline, meta, df)
        out_path = Path(
            cfg.get("output") or Path(cfg.paths.run_reports) / "predict" / "predictions.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info("Wrote %d predictions -> %s", len(out), out_path)
        what = str(inp) if inp else f"next {len(out)} periods"
        track_prediction(
            cfg,
            {"input": what, "weights": str(cfg.paths.run_model), "task": meta["task"]},
            out_path,
            out,
        )
        return
    if df is None:
        raise ValueError("Give the rows to predict: +input=<file.csv>")
    df_in = df.drop(columns=[meta.get("target") or cfg.data.get("target")], errors="ignore")
    pred = pipeline.predict(df_in)
    out = df.copy()
    out["prediction"] = [class_names[int(i)] for i in pred]
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df_in)
        for i, c in enumerate(class_names):
            out[f"proba_{c}"] = proba[:, i]
    out_path = Path(
        cfg.get("output") or Path(cfg.paths.run_reports) / "predict" / "predictions.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Wrote %d predictions -> %s", len(out), out_path)
    track_prediction(
        cfg,
        {"input": str(inp), "weights": model_uri or str(cfg.paths.run_model)},
        out_path,
        out,
    )


if __name__ == "__main__":
    main()
