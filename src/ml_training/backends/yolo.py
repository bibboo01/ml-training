"""Ultralytics YOLO: detect / segment / classify / pose / obb (detection_data + modeling/yolo)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from ml_training.detection_data import featurize, prepare

__all__ = ["featurize", "predict", "prepare", "train"]


def train(cfg: DictConfig) -> dict[str, float]:
    from ml_training.modeling.yolo import train_yolo

    return train_yolo(cfg)


def predict(cfg: DictConfig) -> Path:
    from ml_training.modeling.predict import predict_yolo

    return predict_yolo(cfg)
