"""Stage 1 – get raw data into ``data/raw/<name>.csv`` (tracked by DVC, never edited by hand).

uv run mlt-prepare                                   # default: breast_cancer
uv run mlt-prepare data=iris
uv run mlt-prepare data=csv data.name=churn data.path=data/external/churn.csv data.target=Churn
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from ml_training import lineage
from ml_training.backends import data_backend
from ml_training.friendly import friendly_errors
from ml_training.i18n import tr
from ml_training.utils import CONFIG_DIR, ensure_dir, log


def load_raw(cfg: DictConfig) -> pd.DataFrame:
    d = cfg.data
    if d.source == "sklearn":
        import sklearn.datasets as skd

        bunch = getattr(skd, d.loader)(as_frame=True)
        df = bunch.frame.copy()
        if d.target != "target" and "target" in df.columns:
            df = df.rename(columns={"target": d.target})
        return df

    if d.source == "openml":
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(
            name=d.loader, version=d.get("version", 1), as_frame=True, parser="auto"
        )
        df = bunch.frame.copy()
        return df

    if d.source == "csv":
        path = Path(d.path)
        if not path.is_absolute():
            path = Path(cfg.paths.root) / path
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")

    if d.source == "sample":  # generated regression / forecasting tables (tabular_samples.py)
        from ml_training.tabular_samples import make

        return make(d.sample)

    raise ValueError(f"Unknown data.source: {d.source!r} (use sklearn | csv | openml | sample)")


def run(cfg: DictConfig) -> Path:
    backend = data_backend(cfg)  # images (YOLO) and other non-tabular data
    if backend:
        return backend.prepare(cfg)
    df = load_raw(cfg)
    if cfg.data.target not in df.columns:
        cols = ", ".join(map(str, df.columns[:40]))
        raise KeyError(
            tr(
                f"ไม่พบคอลัมน์ target '{cfg.data.target}' – คอลัมน์ที่มี: {cols}",
                f"Target column '{cfg.data.target}' not found – columns: {cols}",
            )
        )
    out = ensure_dir(cfg.paths.raw) / f"{cfg.data.name}.csv"
    df.to_csv(out, index=False)
    log.info("Saved raw data %s  shape=%s", out, df.shape)
    return out


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
@friendly_errors
def main(cfg: DictConfig) -> None:
    out = run(cfg)
    lineage.record("prepare", lineage.history_path(cfg), dataset=cfg.data.name, output=str(out))


if __name__ == "__main__":
    main()
