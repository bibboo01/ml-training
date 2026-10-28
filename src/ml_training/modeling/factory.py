"""Model abstraction.

Every model – scikit-learn, XGBoost/LightGBM, or our PyTorch Lightning wrapper – is used
through the scikit-learn estimator API (``fit / predict / predict_proba``). It is wrapped in a
``Pipeline(preprocess -> model)`` so preprocessing is learned from training data only and is
saved together with the model.

To add a new model, create ``configs/model/<name>.yaml`` with a ``class_path`` – no code needed
(+ ``regressor:`` = the class used when the target is a number: regression / forecasting).
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


def estimator_path(model_cfg: DictConfig, task: str = "classify") -> str:
    """class_path for classification, regressor for regress / forecast."""
    if task == "classify":
        return str(model_cfg.class_path)
    if not model_cfg.get("regressor"):
        raise ValueError(
            f"Model {model_cfg.name} cannot predict numbers – add `regressor:` to its config "
            "or use random_forest / gradient_boosting / lightgbm / xgboost"
        )
    return str(model_cfg.regressor)


def search_space(model_cfg: DictConfig, task: str = "classify"):
    if task != "classify" and model_cfg.get("regressor_search_space"):
        return model_cfg.regressor_search_space
    return model_cfg.get("search_space")


def build_model(
    model_cfg: DictConfig,
    feature_types: dict[str, list[str]],
    overrides: dict[str, Any] | None,
    task: str = "classify",
) -> Pipeline:
    params = OmegaConf.to_container(model_cfg.params, resolve=True) or {}
    if task != "classify" and model_cfg.get("regressor_params"):
        params.update(OmegaConf.to_container(model_cfg.regressor_params, resolve=True))
    params.update(overrides or {})
    estimator_cls = import_class(estimator_path(model_cfg, task))
    estimator = estimator_cls(**_filter_params(estimator_cls, params))
    if (
        task != "classify"
    ):  # learn a standardised target: SVR / Ridge / MLP need it, trees don't mind
        from sklearn.compose import TransformedTargetRegressor

        estimator = TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    return Pipeline([("preprocess", build_preprocessor(feature_types)), ("model", estimator)])


def _filter_params(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Drop params the estimator does not accept (e.g. random_state on some models)."""
    import inspect

    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return params  # takes **kwargs (LightGBM, XGBoost): pass everything through
    try:
        valid = cls().get_params().keys()
    except Exception:
        return params
    return {k: v for k, v in params.items() if k in valid}
