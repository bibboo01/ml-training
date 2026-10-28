"""Time-series forecasting as tabular regression: lag / rolling / calendar features.

Each row predicts the target at its own time from earlier values only (lags start at 1), so any
regressor (LightGBM, XGBoost, random forest ...) becomes a forecaster. Several series in one
file (e.g. one per shop / product) are handled with ``data.group_column``.
Future values are forecast step by step: every prediction becomes the newest lag of the next.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

TIME_NAMES = ("date", "datetime", "time", "timestamp", "ds", "period", "month", "day", "วันที่")


def find_time_column(df: pd.DataFrame, given: str | None = None) -> str:
    if given:
        if given not in df.columns:
            raise KeyError(f"time_column {given!r} not found. Columns: {list(df.columns)[:20]}")
        return given
    by_name = {c.lower(): c for c in df.columns}
    for n in TIME_NAMES:
        if n in by_name:
            return by_name[n]
    for c in df.columns:  # first text column that parses as dates
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            with warnings.catch_warnings():  # "could not infer format" for non-date text
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[c].head(50), errors="coerce")
            if parsed.notna().mean() > 0.9:
                return c
    raise ValueError("No date / time column found – set data.time_column=<column name>")


def calendar(t: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=t.index)
    out["year"], out["month"], out["day"] = t.dt.year, t.dt.month, t.dt.day
    out["dayofweek"], out["dayofyear"] = t.dt.dayofweek, t.dt.dayofyear
    out["weekofyear"] = t.dt.isocalendar().week.astype(int).to_numpy()
    out["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    if (t.dt.hour != 0).any() or (t.dt.minute != 0).any():
        out["hour"], out["minute"] = t.dt.hour, t.dt.minute
    return out


def make_features(
    df: pd.DataFrame,
    time_col: str,
    target: str,
    group_col: str | None,
    lags: list[int],
    rolling: list[int],
) -> pd.DataFrame:
    """Sorted copy of df + calendar, lag_<k> and roll_mean / roll_std_<w> columns of the target."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    keys = [group_col, time_col] if group_col else [time_col]
    df = df.sort_values(keys).reset_index(drop=True)
    y = pd.to_numeric(df[target], errors="coerce")
    g = y.groupby(df[group_col]) if group_col else None
    feats = calendar(df[time_col])
    for k in lags:
        feats[f"lag_{k}"] = g.shift(k) if g is not None else y.shift(k)
    prev = g.shift(1) if g is not None else y.shift(1)
    for w in rolling:
        r = (
            prev.groupby(df[group_col]).rolling(w, min_periods=1)
            if group_col
            else prev.rolling(w, min_periods=1)
        )
        mean, std = r.mean(), r.std()
        if group_col:
            mean, std = mean.reset_index(level=0, drop=True), std.reset_index(level=0, drop=True)
        feats[f"roll_mean_{w}"], feats[f"roll_std_{w}"] = mean.sort_index(), std.sort_index()
    return pd.concat([df, feats.drop(columns=[c for c in feats if c in df])], axis=1)


def infer_freq(times: pd.Series) -> str | None:
    t = pd.Series(pd.to_datetime(times).unique()).sort_values()
    if len(t) < 3:
        return None
    freq = pd.infer_freq(t.tail(50))
    if freq:
        return freq
    step = t.diff().median()
    return pd.tseries.frequencies.to_offset(step).freqstr if pd.notna(step) else None


def forecast(pipeline, history: pd.DataFrame, meta: dict[str, Any], steps: int) -> pd.DataFrame:
    """Recursive forecast of `steps` future periods after the end of `history` (raw rows).

    Other columns (exogenous features) keep their last known value per series."""
    ts = meta["timeseries"]
    time_col, target, group_col = ts["time_column"], meta["target"], ts.get("group_column")
    lags, rolling = list(ts["lags"]), list(ts["rolling"])
    freq = ts.get("freq") or infer_freq(history[time_col])
    if not freq:
        raise ValueError(
            "Cannot tell the time step of the data (daily, monthly ...) – too few rows"
        )
    offset = pd.tseries.frequencies.to_offset(freq)
    hist = history.copy()
    hist[time_col] = pd.to_datetime(hist[time_col])
    keep = max(lags + rolling) + 1
    groups = list(hist[group_col].dropna().unique()) if group_col else [None]
    out = []
    for grp in groups:
        h = hist[hist[group_col] == grp] if group_col else hist
        h = h.sort_values(time_col).tail(keep).reset_index(drop=True)
        for _ in range(steps):
            nxt = h.iloc[[-1]].copy()
            nxt[time_col] = nxt[time_col] + offset
            nxt[target] = np.nan
            h = pd.concat([h, nxt], ignore_index=True)
            feats = make_features(h, time_col, target, group_col, lags, rolling).iloc[[-1]]
            pred = float(pipeline.predict(feats)[0])
            h.loc[h.index[-1], target] = pred
            row = {time_col: h[time_col].iloc[-1], "forecast": pred}
            if group_col:
                row = {group_col: grp, **row}
            out.append(row)
            h = h.tail(keep).reset_index(drop=True)
    return pd.DataFrame(out)
