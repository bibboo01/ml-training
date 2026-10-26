"""Stage 2 – clean, encode the target and split into train / val / test.

Output: ``data/processed/<name>/{train,val,test}.csv`` + ``meta.json``.
Feature scaling / encoding is NOT done here – it lives inside the model pipeline so it is
fitted on the training data only (no leakage) and travels with the saved model.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from ml_training.utils import CONFIG_DIR, ensure_dir, log, read_json, write_json

TARGET_COL = "__target__"


def infer_feature_types(X: pd.DataFrame) -> dict[str, list[str]]:
    numeric = X.select_dtypes(include="number").columns.tolist()
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric or c in bool_cols]
    numeric = [c for c in numeric if c not in bool_cols]
    return {"numeric": numeric, "categorical": categorical}


def run(cfg: DictConfig) -> Path:
    d = cfg.data
    raw_path = Path(cfg.paths.raw) / f"{d.name}.csv"
    df = pd.read_csv(raw_path)
    df = df.drop(columns=list(d.drop_columns or []), errors="ignore")
    df = df.dropna(subset=[d.target]).drop_duplicates()

    # Encode the target as 0..K-1 and remember the original labels.
    y_raw = df.pop(d.target)
    uniq = y_raw.unique().tolist()
    classes = sorted(uniq) if pd.api.types.is_numeric_dtype(y_raw) else sorted(uniq, key=str)
    mapping = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(mapping).astype(int)

    class_names = list(d.class_names) if d.get("class_names") else [str(c) for c in classes]
    if len(class_names) != len(classes):
        log.warning("class_names length mismatch – falling back to raw labels")
        class_names = [str(c) for c in classes]

    strat = y if d.stratify else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        df, y, test_size=d.test_size, random_state=cfg.seed, stratify=strat
    )
    strat = y_tmp if d.stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=d.val_size, random_state=cfg.seed, stratify=strat
    )

    out_dir = ensure_dir(cfg.paths.processed)
    for split, X_, y_ in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        X_.assign(**{TARGET_COL: y_.values}).to_csv(out_dir / f"{split}.csv", index=False)

    meta = {
        "dataset": d.name,
        "target": d.target,
        "classes": [str(c) for c in classes],
        "class_names": class_names,
        "n_classes": len(classes),
        "features": infer_feature_types(df),
        "n_rows": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "data_config": OmegaConf.to_container(d, resolve=True),
    }
    write_json(meta, out_dir / "meta.json")
    log.info("Processed %s -> %s  %s", raw_path.name, out_dir, meta["n_rows"])
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
    run(cfg)


if __name__ == "__main__":
    main()
