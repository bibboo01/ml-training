"""HuggingFace transformers: text classification, NER and documents (text_data + modeling/text)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from ml_training.text_data import featurize, prepare

__all__ = ["featurize", "predict", "prepare", "train"]


def train(cfg: DictConfig) -> dict[str, float]:
    from ml_training.modeling.text import train_text

    return train_text(cfg)


def predict(cfg: DictConfig) -> Path:
    from ml_training.modeling.text import predict_text

    return predict_text(cfg)
