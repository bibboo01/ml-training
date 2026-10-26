"""Model abstraction.

Every model – scikit-learn, XGBoost/LightGBM, or our PyTorch Lightning wrapper – is used
through the scikit-learn estimator API (``fit / predict / predict_proba``). It is wrapped in a
``Pipeline(preprocess -> model)`` so preprocessing is learned from training data only and is
saved together with the model.

To add a new model, create ``configs/model/<name>.yaml`` with a ``class_path`` – no code needed.
"""

from __future__ import annotations

import importlib
from typing import Any

from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def import_class(class_path: str) -> type:
    module, _, name = class_path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def build_preprocessor(feature_types: dict[str, list[str]]) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50),
            ),
        ]
    )
    transformers = []
    if feature_types.get("numeric"):
        transformers.append(("num", numeric, feature_types["numeric"]))
    if feature_types.get("categorical"):
        transformers.append(("cat", categorical, feature_types["categorical"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_model(
    model_cfg: DictConfig, feature_types: dict[str, list[str]], overrides: dict[str, Any] | None
) -> Pipeline:
    params = OmegaConf.to_container(model_cfg.params, resolve=True) or {}
    params.update(overrides or {})
    estimator_cls = import_class(model_cfg.class_path)
    estimator = estimator_cls(**_filter_params(estimator_cls, params))
    return Pipeline([("preprocess", build_preprocessor(feature_types)), ("model", estimator)])


def _filter_params(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Drop params the estimator does not accept (e.g. random_state on some models)."""
    try:
        valid = cls().get_params().keys()
    except Exception:
        return params
    return {k: v for k, v in params.items() if k in valid}
