"""Batch prediction with a trained model.

uv run mlt-predict +input=data/external/new_rows.csv
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

from ml_training.utils import CONFIG_DIR, log, sqlite_uri


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
    (pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source", "class", "conf"])).to_csv(
        out_path, index=False
    )
    log.info("%d predictions -> %s  (annotated files in %s)", len(rows), out_path, out_dir)
    return out_path


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.model.backend == "ultralytics":
        predict_yolo(cfg)
        return
    root = Path(cfg.paths.root)
    inp = Path(cfg.input)
    df = pd.read_csv(inp if inp.is_absolute() else root / inp)

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

    df_in = df.drop(columns=[cfg.data.target], errors="ignore")
    pred = pipeline.predict(df_in)
    out = df.copy()
    out["prediction"] = [class_names[int(i)] for i in pred]
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df_in)
        for i, c in enumerate(class_names):
            out[f"proba_{c}"] = proba[:, i]
    out_path = Path(cfg.get("output") or root / "reports" / "predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info("Wrote %d predictions -> %s", len(out), out_path)


if __name__ == "__main__":
    main()
