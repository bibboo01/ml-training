"""Compare every trained model of one dataset (reads reports/<dataset>/*/metrics.json).

uv run python -m ml_training.compare helmet          # table + reports/helmet/compare.xlsx
uv run python -m ml_training.compare helmet --best   # prints only the best model name
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ml_training.utils import PROJECT_ROOT

# first metric found is the ranking key (detection, then classification)
RANK_KEYS = ["map50_95", "f1", "accuracy"]
COLUMNS = [
    "map50", "map50_95", "precision", "recall", "f1", "accuracy",
    "TP", "FP", "FN", "TN", "fit_seconds",
]  # fmt: skip


def collect(dataset: str, reports: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((reports / dataset).glob("*/metrics.json")):
        test = json.loads(path.read_text(encoding="utf-8")).get("test") or {}
        weights = PROJECT_ROOT / "models" / dataset / f"{path.parent.name}.pt"
        rows.append({"model": path.parent.name, **test, "has_pt": weights.exists()})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    key = next((k for k in RANK_KEYS if k in df), None)
    if key:
        df = df.sort_values(key, ascending=False, na_position="last")
    cols = ["model"] + [c for c in COLUMNS if c in df]
    return df[cols + ["has_pt"]].reset_index(drop=True)


def write_excel(df: pd.DataFrame, path: Path, dataset: str) -> None:
    bounded = [c for c in ["map50", "map50_95", "precision", "recall", "f1", "accuracy"] if c in df]
    out = df.drop(columns="has_pt")
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        out.to_excel(xw, sheet_name="Compare", index=False)
        wb, ws = xw.book, xw.sheets["Compare"]
        ws.set_column(0, 0, 18)
        ws.set_column(1, len(out.columns), 12, wb.add_format({"num_format": "0.000"}))
        if bounded:
            chart = wb.add_chart({"type": "column"})
            n = len(out)
            for metric in bounded:
                col = out.columns.get_loc(metric)
                chart.add_series(
                    {
                        "name": ["Compare", 0, col],
                        "categories": ["Compare", 1, 0, n, 0],
                        "values": ["Compare", 1, col, n, col],
                    }
                )
            chart.set_title({"name": f"{dataset}: model comparison"})
            chart.set_y_axis({"min": 0, "max": 1})
            chart.set_size({"width": 760, "height": 400})
            ws.insert_chart(n + 3, 0, chart)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset")
    ap.add_argument("--best", action="store_true", help="print only the best model with a .pt")
    ap.add_argument("--reports", default=str(PROJECT_ROOT / "reports"))
    args = ap.parse_args(argv)
    reports = Path(args.reports)
    df = collect(args.dataset, reports)
    if df.empty:
        print(f"No trained models found in {reports / args.dataset}", file=sys.stderr)
        return 1
    if args.best:
        ok = df[df["has_pt"]]
        print((ok if len(ok) else df)["model"].iloc[0])
        return 0
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(df.drop(columns="has_pt").to_string(index=False))
    xlsx = reports / args.dataset / "compare.xlsx"
    write_excel(df, xlsx, args.dataset)
    print(f"\nBest: {df['model'].iloc[0]}   Excel: {xlsx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
