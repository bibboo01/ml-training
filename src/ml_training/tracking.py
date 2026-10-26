"""MLflow helpers shared by every training backend."""

from __future__ import annotations

import mlflow
from omegaconf import DictConfig

from ml_training.utils import file_uri, sqlite_uri


def setup_mlflow(cfg: DictConfig) -> str:
    mlflow.set_tracking_uri(sqlite_uri(cfg.mlflow.tracking_uri))
    exp = mlflow.get_experiment_by_name(cfg.experiment_name)
    if exp is None:
        return mlflow.create_experiment(
            cfg.experiment_name, artifact_location=file_uri(cfg.mlflow.artifact_location)
        )
    return exp.experiment_id


def log_params(params: dict) -> None:
    mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
