"""Stage 2 – clean, encode the target and split into train / val / test.

Tabular tasks (data.task: auto): classify (labels), regress (a number, e.g. price) and
forecast (a number over time: lag / rolling / calendar features, split in time order).

Output: ``data/processed/<name>/{train,val,test}.csv`` + ``meta.json``.
Feature scaling / encoding is NOT done here – it lives inside the model pipeline so it is
fitted on the training data only (no leakage) and travels with the saved model.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from ml_training import lineage
from ml_training.backends import data_backend
from ml_training.utils import CONFIG_DIR, ensure_dir, log, read_json, write_json

TARGET_COL = "__target__"
TASKS = ("classify", "regress", "forecast")


def infer_feature_types(X: pd.DataFrame) -> dict[str, list[str]]:
    numeric = X.select_dtypes(include="number").columns.tolist()
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric or c in bool_cols]
    numeric = [c for c in numeric if c not in bool_cols]
    return {"numeric": numeric, "categorical": categorical}


def detect_task(y: pd.Series, d: DictConfig) -> str:
    """classify | regress | forecast (data.task: auto decides from the target and time column)."""
    task = str(d.get("task") or "auto")
    if task != "auto":
        if task not in TASKS:
            raise ValueError(f"data.task must be auto or one of {TASKS} – got {task!r}")
        return task
    if d.get("time_column"):
        return "forecast"
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    return "regress" if numeric and y.nunique() > 20 else "classify"


def _chrono_split(df: pd.DataFrame, time_col: str, test_size: float, val_size: float):
    """Time order, never shuffled: train = oldest, then val, test = newest periods."""
    times = np.sort(df[time_col].unique())
    t_test = times[int(len(times) * (1 - test_size))]
    rest = times[times < t_test]
    t_val = rest[int(len(rest) * (1 - val_size))]
    return (
        df[df[time_col] < t_val],
        df[(df[time_col] >= t_val) & (df[time_col] < t_test)],
        df[df[time_col] >= t_test],
    )


def run(cfg: DictConfig) -> Path:
    backend = data_backend(cfg)  # images (YOLO) and other non-tabular data
    if backend:
        return backend.featurize(cfg)
    d = cfg.data
    raw_path = Path(cfg.paths.raw) / f"{d.name}.csv"
    df = pd.read_csv(raw_path)
    df = df.drop(columns=list(d.drop_columns or []), errors="ignore")
    df = df.dropna(subset=[d.target])
    task = detect_task(df[d.target], d)
    extra: dict = {}
    if task == "forecast":
        from ml_training import timeseries as ts

        time_col = ts.find_time_column(df, d.get("time_column"))
        group_col = d.get("group_column")
        lags = [int(x) for x in (d.get("lags") or [1, 2, 3, 7, 14])]
        rolling = [int(x) for x in (d.get("rolling") or [7, 28])]
        df[d.target] = pd.to_numeric(df[d.target], errors="coerce")
        df = ts.make_features(df, time_col, d.target, group_col, lags, rolling)
        df = df.dropna(subset=[f"lag_{max(lags)}", d.target])
        extra["timeseries"] = {
            "time_column": time_col,
            "group_column": group_col,
            "lags": lags,
            "rolling": rolling,
            "freq": ts.infer_freq(df[time_col]),
            "last_time": str(df[time_col].max()),
        }
        df[time_col] = df[time_col].astype(str)
    else:
        df = df.drop_duplicates()

    y_raw = df.pop(d.target)
    if task == "classify":  # encode the target as 0..K-1 and remember the original labels
        uniq = y_raw.unique().tolist()
        classes = sorted(uniq) if pd.api.types.is_numeric_dtype(y_raw) else sorted(uniq, key=str)
        mapping = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(mapping).astype(int)
        class_names = list(d.class_names) if d.get("class_names") else [str(c) for c in classes]
        if len(class_names) != len(classes):
            log.warning("class_names length mismatch – falling back to raw labels")
            class_names = [str(c) for c in classes]
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(float)
        classes, class_names = [], []

    if task == "forecast":
        time_col = extra["timeseries"]["time_column"]
        parts = _chrono_split(
            df.assign(**{TARGET_COL: y.values}), time_col, d.test_size, d.val_size
        )
        splits = [
            (n, p.drop(columns=TARGET_COL), p[TARGET_COL])
            for n, p in zip(("train", "val", "test"), parts, strict=True)
        ]
    else:
        strat = y if (d.stratify and task == "classify") else None
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            df, y, test_size=d.test_size, random_state=cfg.seed, stratify=strat
        )
        strat = y_tmp if (d.stratify and task == "classify") else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=d.val_size, random_state=cfg.seed, stratify=strat
        )
        splits = [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]

    out_dir = ensure_dir(cfg.paths.processed)
    for split, X_, y_ in splits:
        X_.assign(**{TARGET_COL: y_.values}).to_csv(out_dir / f"{split}.csv", index=False)

    features = df
    if task == "forecast":  # the time stamp itself is not a feature (calendar columns are)
        features = df.drop(columns=[extra["timeseries"]["time_column"]])
    meta = {
        "dataset": d.name,
        "task": task,
        "target": d.target,
        "classes": [str(c) for c in classes],
        "class_names": class_names,
        "n_classes": len(classes),
        "features": infer_feature_types(features),
        "n_rows": {s: len(X_) for s, X_, _ in splits},
        "data_config": OmegaConf.to_container(d, resolve=True),
        **extra,
    }
    write_json(meta, out_dir / "meta.json")
    log.info("Processed %s -> %s  task=%s  %s", raw_path.name, out_dir, task, meta["n_rows"])
    return out_dir


def load_splits(processed_dir: str | Path):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test, meta)."""
    processed_dir = Path(processed_dir)
    meta = read_json(processed_dir / "meta.json")
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(processed_dir / f"{split}.csv")
        y = df.pop(TARGET_COL).to_numpy()
        out += [df, y]
    return (*out, meta)


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    out = run(cfg)
    meta = read_json(out / "meta.json") if (out / "meta.json").exists() else {}
    lineage.record(
        "featurize",
        lineage.history_path(cfg),
        dataset=cfg.data.name,
        task=meta.get("task", "tabular"),
        rows=meta.get("n_rows"),
        classes=meta.get("n_classes"),
        output=str(out),
    )


if __name__ == "__main__":
    main()
