"""Image classification (timm: ResNet, EfficientNet, ConvNeXt, ViT ...) and semantic
segmentation (SegFormer) – vision_data + modeling/vision."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from ml_training.vision_data import featurize, prepare

__all__ = ["featurize", "predict", "prepare", "train"]


def train(cfg: DictConfig) -> dict[str, float]:
    from ml_training.modeling.vision import train_vision

    return train_vision(cfg)


def predict(cfg: DictConfig) -> Path:
    from ml_training.modeling.vision import predict_vision

    return predict_vision(cfg)
