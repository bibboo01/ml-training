"""Plug-in trainers: one backend per kind of data / model family.

A backend is a module with four functions, each taking the Hydra config:
    prepare(cfg)    raw data -> data/raw/...            (mlt-prepare)
    featurize(cfg)  -> data/processed/<name>/ + meta.json (mlt-featurize)
    train(cfg)      -> model + MLflow run + Excel report  (mlt-train)
    predict(cfg)    -> predictions CSV + MLflow run       (mlt-predict)

The data side is picked from the data config (data.kind, or data.source for older configs),
the model side from the model config (model.backend). Tabular scikit-learn / Lightning models
are the built-in default (no backend module). To add a new family of models, write
backends/<name>.py and add it to DATA_KINDS / MODEL_BACKENDS below.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from omegaconf import DictConfig

# data.kind (or data.source) -> backend module
DATA_KINDS = {"yolo": "yolo", "ultralytics": "yolo", "text": "text"}
# model.backend -> backend module
MODEL_BACKENDS = {"ultralytics": "yolo", "transformers": "text"}


def data_backend(cfg: DictConfig) -> ModuleType | None:
    """Backend that prepares this dataset, or None for tabular data."""
    d = cfg.data
    name = DATA_KINDS.get(str(d.get("kind") or d.get("source") or ""))
    return import_module(f"ml_training.backends.{name}") if name else None


def model_backend(cfg: DictConfig) -> ModuleType | None:
    """Backend that trains / predicts with this model, or None for tabular models."""
    name = MODEL_BACKENDS.get(str(cfg.model.get("backend") or ""))
    return import_module(f"ml_training.backends.{name}") if name else None
