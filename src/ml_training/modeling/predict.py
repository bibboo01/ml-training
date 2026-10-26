"""Batch prediction with a trained model.

uv run mlt-predict +input=data/external/new_rows.csv
uv run mlt-predict +input=new.csv +output=preds.csv \
    +model_uri=models:/breast_cancer-random_forest/latest
"""

from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig

from ml_training.utils import CONFIG_DIR, log, sqlite_uri


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
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
