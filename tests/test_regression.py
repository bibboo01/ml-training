"""Regression + time-series forecasting on the generated table samples."""

import json

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ml_training import dataset, features, timeseries
from ml_training.metrics import regression_metrics
from ml_training.modeling import train
from ml_training.modeling.factory import build_model
from ml_training.utils import CONFIG_DIR


def _cfg(tmp_path, *overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                f"paths.root={tmp_path.as_posix()}",
                "mlflow.register_model=false",
                *overrides,
            ],
        )


def test_detect_task():
    d = OmegaConf.create({"task": "auto"})
    assert features.detect_task(pd.Series(np.linspace(0, 1, 100)), d) == "regress"
    assert features.detect_task(pd.Series([0, 1] * 50), d) == "classify"
    assert features.detect_task(pd.Series(["a", "b"] * 50), d) == "classify"
    assert (
        features.detect_task(pd.Series(range(100)), OmegaConf.create({"time_column": "d"}))
        == "forecast"
    )


def test_regression_metrics():
    m = regression_metrics([10, 20, 30, 40], [12, 18, 30, 44], naive=[np.nan, 10, 20, 30])
    assert m["mae"] == pytest.approx(2.0) and m["rmse"] == pytest.approx(np.sqrt(6))
    assert m["within_10pct"] == pytest.approx(0.75) and m["bias"] == pytest.approx(1.0)
    assert m["mase"] == pytest.approx((2 + 0 + 4) / 3 / 10)


def test_lag_features_per_series_and_no_leak():
    df = pd.DataFrame(
        {
            "date": list(pd.date_range("2024-01-01", periods=5).astype(str)) * 2,
            "shop": ["a"] * 5 + ["b"] * 5,
            "y": list(range(5)) + list(range(100, 105)),
        }
    )
    f = timeseries.make_features(df, "date", "y", "shop", [1, 2], [3])
    b = f[f.shop == "b"].reset_index(drop=True)
    assert b.lag_1.tolist()[1:] == [100, 101, 102, 103] and np.isnan(b.lag_1[0])  # not from shop a
    assert b.roll_mean_3[3] == pytest.approx(101.0)  # mean of 100, 101, 102 – today excluded
    assert f.dayofweek.notna().all()


def test_regressor_and_classifier_from_one_config(tmp_path):
    cfg = _cfg(tmp_path, "model=lightgbm")
    feats = {"numeric": ["x"], "categorical": []}
    reg = build_model(cfg.model, feats, {}, "regress")
    assert "LGBMRegressor" in repr(reg.named_steps["model"].regressor)
    assert "LGBMClassifier" in repr(build_model(cfg.model, feats, {}).named_steps["model"])
    with pytest.raises(ValueError, match="cannot predict numbers"):
        build_model(_cfg(tmp_path, "model=lightning_mlp").model, feats, {}, "regress")


def test_regression_end_to_end(tmp_path):
    cfg = _cfg(tmp_path, "data=house-prices", "model=gradient_boosting", "hpo.n_trials=2")
    dataset.run(cfg)
    features.run(cfg)
    s = train.train(cfg)
    assert s["r2"] > 0.7
    assert (tmp_path / "reports/house-prices/gradient_boosting/report.xlsx").exists()


def test_forecast_end_to_end(tmp_path):
    cfg = _cfg(
        tmp_path,
        "data=shop-sales",
        "model=random_forest",
        "hpo=none",
        "model.params.n_estimators=50",
    )
    dataset.run(cfg)
    out = features.run(cfg)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == "forecast" and meta["timeseries"]["freq"] == "D"
    tr, te = pd.read_csv(out / "train.csv"), pd.read_csv(out / "test.csv")
    assert tr.date.max() < te.date.min()  # split in time order
    s = train.train(cfg)
    assert s["r2"] > 0.8 and s["mase"] < 1  # better than "same as yesterday"
    import joblib

    bundle = joblib.load(cfg.paths.run_model)
    raw = pd.read_csv(tmp_path / "data/raw/shop-sales.csv")
    fc = timeseries.forecast(bundle["pipeline"], raw, bundle["meta"], 5)
    assert len(fc) == 15 and fc.date.min() == pd.Timestamp("2025-01-01") and (fc.forecast > 0).all()
