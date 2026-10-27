"""Excel report with native (editable) Excel charts via XlsxWriter.

Sheets: Summary · Confusion · PerClass · ROC · HPO_Trials · ParamImportance · Predictions · Config
Detection runs add: Dataset · TrainingHistory · Plots (Ultralytics PNGs).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_training.metrics import EvalResult

BOUNDED_METRICS = [
    "map50",
    "map50_95",
    "map75",
    "box_map50",
    "box_map50_95",
    "top1",
    "top5",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
]
PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#7C3AED", "#0891B2", "#DB2777", "#65A30D"]


def _fmt(wb):
    return {
        "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#1F2937"}),
        "h": wb.add_format(
            {"bold": True, "bg_color": "#1F2937", "font_color": "white", "border": 1}
        ),
        "key": wb.add_format({"bold": True, "bg_color": "#F3F4F6", "border": 1}),
        "cell": wb.add_format({"border": 1}),
        "num": wb.add_format({"border": 1, "num_format": "0.0000"}),
        "int": wb.add_format({"border": 1, "num_format": "0"}),
        "big": wb.add_format(
            {"bold": True, "font_size": 14, "align": "center", "valign": "vcenter", "border": 2}
        ),
        "tp": wb.add_format(
            {"bold": True, "font_size": 14, "align": "center", "border": 2, "bg_color": "#DCFCE7"}
        ),
        "bad": wb.add_format(
            {"bold": True, "font_size": 14, "align": "center", "border": 2, "bg_color": "#FEE2E2"}
        ),
    }


def _write_df(ws, df: pd.DataFrame, row: int, col: int, f: dict, index: bool = False) -> tuple:
    """Write a dataframe as a bordered table. Returns (first_row, last_row, first_col, last_col)."""
    if index:
        df = df.reset_index()
    for j, name in enumerate(df.columns):
        ws.write(row, col + j, str(name), f["h"])
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(rec):
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                ws.write_blank(row + i, col + j, None, f["cell"])
            elif isinstance(v, bool | np.bool_):
                ws.write(row + i, col + j, str(v), f["cell"])
            elif isinstance(v, int | np.integer):
                ws.write_number(row + i, col + j, v, f["int"])
            elif isinstance(v, float):
                ws.write_number(row + i, col + j, v, f["num"])
            else:
                ws.write(row + i, col + j, str(v), f["cell"])
    for j, name in enumerate(df.columns):
        width = max([len(str(name))] + [len(str(x)) for x in df.iloc[:50, j]]) + 2
        ws.set_column(col + j, col + j, min(max(width, 9), 45))
    return row, row + len(df), col, col + len(df.columns) - 1


def write_report(
    path: str | Path,
    run_info: dict[str, Any],
    result: EvalResult,
    best_params: dict[str, Any],
    trials: pd.DataFrame | None = None,
    importances: dict[str, float] | None = None,
    predictions: pd.DataFrame | None = None,
    config_flat: dict[str, Any] | None = None,
    val_summary: dict[str, float] | None = None,
    history: pd.DataFrame | None = None,
    images: list[Path] | None = None,
    dataset: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        wb = xw.book
        f = _fmt(wb)
        _summary_sheet(wb, f, run_info, result, best_params, val_summary)
        _confusion_sheet(wb, f, result)
        _per_class_sheet(wb, f, result)
        if result.roc:
            _roc_sheet(wb, f, result)
        if dataset is not None and len(dataset):
            _dataset_sheet(wb, f, dataset)
        if history is not None and len(history):
            _history_sheet(wb, f, history)
        if images:
            _images_sheet(wb, f, images)
        if trials is not None and len(trials):
            _trials_sheet(wb, f, trials, run_info.get("hpo_metric", "objective"))
        if importances:
            _importance_sheet(wb, f, importances)
        if predictions is not None:
            ws = wb.add_worksheet("Predictions")
            r0, r1, c0, c1 = _write_df(ws, predictions, 0, 0, f)
            ws.autofilter(r0, c0, r1, c1)
            ws.freeze_panes(1, 0)
        if config_flat:
            ws = wb.add_worksheet("Config")
            cfg_df = pd.DataFrame(
                {"key": list(config_flat), "value": [str(v) for v in config_flat.values()]}
            )
            _write_df(ws, cfg_df, 0, 0, f)
    return path


def _summary_sheet(wb, f, run_info, result: EvalResult, best_params, val_summary):
    ws = wb.add_worksheet("Summary")
    ws.write(0, 0, "ML Training Report", f["title"])
    info = pd.DataFrame({"field": list(run_info), "value": [str(v) for v in run_info.values()]})
    _write_df(ws, info, 2, 0, f)

    row = 4 + len(info)
    # Metrics in [0, 1] first (these go in the chart), then mcc / log_loss / timing.
    bounded = [k for k in BOUNDED_METRICS if k in result.summary]
    other = [k for k in result.summary if k not in bounded and k not in ("TP", "FP", "FN", "TN")]
    metrics = {k: result.summary[k] for k in bounded + other}
    mdf = pd.DataFrame({"metric": list(metrics), "test": list(metrics.values())})
    if val_summary:
        mdf["validation (best trial)"] = [val_summary.get(k) for k in metrics]
    r0, r1, _, _ = _write_df(ws, mdf, row, 0, f)

    chart = wb.add_chart({"type": "bar"})
    chart.add_series(
        {
            "name": "Test",
            "categories": ["Summary", r0 + 1, 0, r0 + len(bounded), 0],
            "values": ["Summary", r0 + 1, 1, r0 + len(bounded), 1],
            "fill": {"color": PALETTE[0]},
            "data_labels": {"value": True, "num_format": "0.000"},
        }
    )
    chart.set_title({"name": "Test metrics"})
    chart.set_x_axis({"min": 0, "max": 1, "num_format": "0.0"})
    chart.set_y_axis({"reverse": True})
    chart.set_legend({"none": True})
    chart.set_size({"width": 620, "height": 380})
    ws.insert_chart(2, 4, chart)

    row = r1 + 2
    ws.write(row, 0, "Best hyperparameters", f["key"])
    bp = pd.DataFrame({"param": list(best_params), "value": [str(v) for v in best_params.values()]})
    if len(bp):
        _write_df(ws, bp, row + 1, 0, f)
    ws.set_column(0, 0, 26)
    ws.set_column(1, 2, 22)


def _confusion_sheet(wb, f, result: EvalResult):
    ws = wb.add_worksheet("Confusion")
    cm = result.confusion
    n = len(cm)
    ws.write(0, 0, "Confusion matrix (rows = true, columns = predicted)", f["title"])
    r0, r1, c0, c1 = _write_df(ws, cm.rename_axis("true \\ pred"), 2, 0, f, index=True)
    ws.conditional_format(
        r0 + 1,
        c0 + 1,
        r1,
        c1,
        {
            "type": "2_color_scale",
            "min_color": "#FFFFFF",
            "max_color": "#2563EB",
        },
    )

    row = r1 + 3
    s = result.summary
    if "TN" not in s:
        label = "all classes (detection/segment/pose/obb: TN is not defined)"
    elif n == 2:
        label = "positive class"
    else:
        label = "sum over classes (one-vs-rest)"
    ws.write(row, 0, f"TP / FP / FN / TN – {label}", f["key"])
    ws.write(row + 1, 1, "Pred +", f["h"])
    ws.write(row + 1, 2, "Pred −", f["h"])
    ws.write(row + 2, 0, "Actual +", f["h"])
    ws.write(row + 3, 0, "Actual −", f["h"])
    ws.write(row + 2, 1, f"TP = {s['TP']}", f["tp"])
    ws.write(row + 2, 2, f"FN = {s['FN']}", f["bad"])
    ws.write(row + 3, 1, f"FP = {s['FP']}", f["bad"])
    ws.write(row + 3, 2, f"TN = {s['TN']}" if "TN" in s else "TN = n/a", f["tp"])
    ws.set_row(row + 2, 30)
    ws.set_row(row + 3, 30)
    ws.set_column(1, max(c1, 2), 16)

    # Stacked column chart: TP/FP/FN/TN per class (data from the PerClass sheet)
    pc_rows = len(result.per_class)
    pc_cols = list(result.per_class.columns)
    chart = wb.add_chart({"type": "column", "subtype": "stacked"})
    for name, color in zip(
        ["TP", "FP", "FN", "TN"], ["#16A34A", "#F59E0B", "#DC2626", "#94A3B8"], strict=True
    ):
        if name not in pc_cols:
            continue
        j = pc_cols.index(name)
        chart.add_series(
            {
                "name": name,
                "categories": ["PerClass", 1, 0, pc_rows, 0],
                "values": ["PerClass", 1, j, pc_rows, j],
                "fill": {"color": color},
            }
        )
    chart.set_title(
        {"name": "TP / FP / FN / TN per class" if "TN" in pc_cols else "TP / FP / FN per class"}
    )
    chart.set_size({"width": 620, "height": 360})
    ws.insert_chart(2, c1 + 2, chart)


def _per_class_sheet(wb, f, result: EvalResult):
    ws = wb.add_worksheet("PerClass")
    pc = result.per_class
    _, r1, _, _ = _write_df(ws, pc, 0, 0, f)
    cols = list(pc.columns)
    chart = wb.add_chart({"type": "column"})
    shown = [
        m for m in ["precision", "recall", "specificity", "f1", "AP50", "AP50_95"] if m in cols
    ]
    for i, m in enumerate(shown):
        j = cols.index(m)
        chart.add_series(
            {
                "name": m,
                "categories": ["PerClass", 1, 0, r1, 0],
                "values": ["PerClass", 1, j, r1, j],
                "fill": {"color": PALETTE[i]},
            }
        )
    chart.set_title({"name": "Per-class metrics"})
    chart.set_y_axis({"min": 0, "max": 1})
    chart.set_size({"width": 720, "height": 380})
    ws.insert_chart(r1 + 2, 0, chart)


def _roc_sheet(wb, f, result: EvalResult):
    ws = wb.add_worksheet("ROC")
    chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
    col = 0
    for i, (name, df) in enumerate(result.roc.items()):
        ws.write(0, col, f"ROC: {name}", f["key"])
        _, r1, _, _ = _write_df(ws, df, 1, col, f)
        chart.add_series(
            {
                "name": str(name),
                "categories": ["ROC", 2, col, r1, col],
                "values": ["ROC", 2, col + 1, r1, col + 1],
                "line": {"color": PALETTE[i % len(PALETTE)], "width": 2},
            }
        )
        col += 4
    # chance diagonal
    ws.write(0, col, "chance", f["key"])
    ws.write_row(1, col, ["fpr", "tpr"], f["h"])
    ws.write_row(2, col, [0, 0])
    ws.write_row(3, col, [1, 1])
    chart.add_series(
        {
            "name": "chance",
            "categories": ["ROC", 2, col, 3, col],
            "values": ["ROC", 2, col + 1, 3, col + 1],
            "line": {"color": "#9CA3AF", "dash_type": "dash"},
        }
    )
    auc = result.summary.get("roc_auc")
    chart.set_title({"name": f"ROC curve (AUC = {auc:.3f})" if auc is not None else "ROC curve"})
    chart.set_x_axis({"name": "False positive rate", "min": 0, "max": 1})
    chart.set_y_axis({"name": "True positive rate", "min": 0, "max": 1})
    chart.set_size({"width": 560, "height": 480})
    ws.insert_chart(1, col + 3, chart)


def _trials_sheet(wb, f, trials: pd.DataFrame, metric: str):
    ws = wb.add_worksheet("HPO_Trials")
    r0, r1, c0, c1 = _write_df(ws, trials, 0, 0, f)
    ws.autofilter(r0, c0, r1, c1)
    ws.freeze_panes(1, 0)
    cols = list(trials.columns)
    num, val, best = cols.index("number"), cols.index("value"), cols.index("best_so_far")

    chart = wb.add_chart({"type": "scatter"})
    chart.add_series(
        {
            "name": f"trial {metric}",
            "categories": ["HPO_Trials", 1, num, r1, num],
            "values": ["HPO_Trials", 1, val, r1, val],
            "marker": {
                "type": "circle",
                "size": 6,
                "fill": {"color": PALETTE[0]},
                "border": {"color": PALETTE[0]},
            },
            "line": {"none": True},
        }
    )
    # One scatter chart, two series: markers for each trial, a line for best-so-far.
    # (XlsxWriter cannot combine two scatter charts.)
    chart.add_series(
        {
            "name": "best so far",
            "categories": ["HPO_Trials", 1, num, r1, num],
            "values": ["HPO_Trials", 1, best, r1, best],
            "line": {"color": PALETTE[1], "width": 2.25},
            "marker": {"type": "none"},
        }
    )
    chart.set_title({"name": "Optuna optimisation history"})
    chart.set_x_axis({"name": "Trial"})
    chart.set_y_axis({"name": metric})
    chart.set_size({"width": 720, "height": 400})
    ws.insert_chart(r1 + 2, 0, chart)


def _importance_sheet(wb, f, importances: dict[str, float]):
    ws = wb.add_worksheet("ParamImportance")
    df = pd.DataFrame({"param": list(importances), "importance": list(importances.values())})
    _, r1, _, _ = _write_df(ws, df, 0, 0, f)
    chart = wb.add_chart({"type": "bar"})
    chart.add_series(
        {
            "name": "importance",
            "categories": ["ParamImportance", 1, 0, r1, 0],
            "values": ["ParamImportance", 1, 1, r1, 1],
            "fill": {"color": PALETTE[4]},
            "data_labels": {"value": True, "num_format": "0.00"},
        }
    )
    chart.set_title({"name": "Hyperparameter importance (fANOVA)"})
    chart.set_y_axis({"reverse": True})
    chart.set_legend({"none": True})
    ws.insert_chart(1, 3, chart)


def _dataset_sheet(wb, f, dataset: pd.DataFrame):
    ws = wb.add_worksheet("Dataset")
    _, r1, _, c1 = _write_df(ws, dataset, 0, 0, f)
    chart = wb.add_chart({"type": "bar", "subtype": "stacked"})
    for j in range(1, c1 + 1):
        chart.add_series(
            {
                "name": ["Dataset", 0, j],
                "categories": ["Dataset", 1, 0, r1, 0],
                "values": ["Dataset", 1, j, r1, j],
                "fill": {"color": PALETTE[(j - 1) % len(PALETTE)]},
            }
        )
    chart.set_title({"name": "Boxes per class and split"})
    chart.set_y_axis({"reverse": True})
    chart.set_size({"width": 620, "height": max(320, 22 * r1 + 120)})
    ws.insert_chart(1, c1 + 2, chart)


def _history_sheet(wb, f, history: pd.DataFrame):
    """Per-epoch curves from Ultralytics results.csv."""
    ws = wb.add_worksheet("TrainingHistory")
    _, r1, _, _ = _write_df(ws, history, 0, 0, f)
    ws.freeze_panes(1, 1)
    cols = list(history.columns)
    x = cols.index("epoch") if "epoch" in cols else 0
    groups = [
        ("Validation metrics per epoch", [c for c in cols if c.startswith("metrics/")], (0, 1)),
        ("Losses per epoch", [c for c in cols if "loss" in c], None),
    ]
    for k, (title, series, yrange) in enumerate(groups):
        if not series:
            continue
        chart = wb.add_chart({"type": "line"})
        for i, c in enumerate(series):
            j = cols.index(c)
            chart.add_series(
                {
                    "name": ["TrainingHistory", 0, j],
                    "categories": ["TrainingHistory", 1, x, r1, x],
                    "values": ["TrainingHistory", 1, j, r1, j],
                    "line": {"color": PALETTE[i % len(PALETTE)], "width": 1.75},
                }
            )
        chart.set_title({"name": title})
        chart.set_x_axis({"name": "epoch"})
        if yrange:
            chart.set_y_axis({"min": yrange[0], "max": yrange[1]})
        chart.set_size({"width": 720, "height": 400})
        ws.insert_chart(r1 + 2 + k * 21, 0, chart)


def _images_sheet(wb, f, images: list[Path]):
    from PIL import Image

    ws = wb.add_worksheet("Plots")
    row = 0
    for img in images:
        with Image.open(img) as im:
            w, h = im.size
        scale = min(1.0, 900 / w)
        ws.write(row, 0, Path(img).name, f["key"])
        ws.insert_image(row + 1, 0, str(img), {"x_scale": scale, "y_scale": scale})
        row += int(h * scale / 20) + 3  # default row height is 20 px
