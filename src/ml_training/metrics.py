"""Classification metrics built on the confusion matrix (TP / FP / FN / TN).

Binary: TP/FP/FN/TN are for the positive class (encoded label 1).
Multiclass: each class gets one-vs-rest TP/FP/FN/TN, plus macro/weighted averages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvalResult:
    summary: dict[str, float]
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    roc: dict[str, pd.DataFrame] = field(default_factory=dict)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def per_class_counts(cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    total = cm.sum()
    rows = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fp - fn)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        rows.append(
            {
                "class": name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "support": tp + fn,
                "precision": precision,
                "recall": recall,
                "specificity": _safe_div(tn, tn + fp),
                "f1": _safe_div(2 * precision * recall, precision + recall),
                "fpr": _safe_div(fp, fp + tn),
                "fnr": _safe_div(fn, fn + tp),
                "npv": _safe_div(tn, tn + fn),
            }
        )
    return pd.DataFrame(rows)


def predict_with_threshold(model, X, n_classes: int, threshold: float = 0.5):
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    if proba is not None and n_classes == 2:
        return (proba[:, 1] >= threshold).astype(int), proba
    return np.asarray(model.predict(X)), proba


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    class_names: list[str],
    average: str = "macro",
) -> EvalResult:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pc = per_class_counts(cm, class_names)
    binary = len(labels) == 2

    if binary and average == "binary":
        row = pc.iloc[1]
        prec, rec, spec, f1 = row.precision, row.recall, row.specificity, row.f1
    elif average == "weighted":
        w = pc.support / max(pc.support.sum(), 1)
        prec, rec, spec, f1 = (
            (pc[c] * w).sum() for c in ["precision", "recall", "specificity", "f1"]
        )
    else:  # macro (also used for "binary" on multiclass)
        prec, rec, spec, f1 = (pc[c].mean() for c in ["precision", "recall", "specificity", "f1"])

    summary: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if binary:
        pos = pc.iloc[1]
        summary.update({k: int(pos[k]) for k in ("TP", "FP", "FN", "TN")})
    else:
        summary.update({k: int(pc[k].sum()) for k in ("TP", "FP", "FN", "TN")})

    roc: dict[str, pd.DataFrame] = {}
    if proba is not None:
        try:
            if binary:
                summary["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
                fpr, tpr, thr = roc_curve(y_true, proba[:, 1])
                roc[class_names[1]] = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
            else:
                summary["roc_auc"] = roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro", labels=labels
                )
                for i, name in enumerate(class_names):
                    fpr, tpr, thr = roc_curve((y_true == i).astype(int), proba[:, i])
                    roc[name] = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
            summary["log_loss"] = log_loss(y_true, proba, labels=labels)
        except ValueError:
            pass  # e.g. a class missing from y_true in a tiny split

    summary = {k: (float(v) if not isinstance(v, int) else v) for k, v in summary.items()}
    confusion = pd.DataFrame(
        cm,
        index=[f"true: {c}" for c in class_names],
        columns=[f"pred: {c}" for c in class_names],
    )
    return EvalResult(summary=summary, per_class=pc, confusion=confusion, roc=roc)


# --------------------------------------------------------------------------- regression
LOWER_IS_BETTER = {"mae", "rmse", "mse", "medae", "mape", "smape", "max_error", "mase"}


def regression_metrics(y_true, y_pred, naive: np.ndarray | None = None) -> dict[str, float]:
    """MAE, RMSE, R², MAPE / sMAPE (in %), median and max error, share within 10 % of the truth.
    naive = the "same as last period" forecast (lag_1) -> MASE < 1 means better than naive."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    nz = np.abs(y) > 1e-12
    denom = np.abs(y) + np.abs(p)
    out = {
        "r2": float(1 - (err**2).sum() / ((y - y.mean()) ** 2).sum())
        if len(y) > 1 and y.std()
        else 0.0,
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "medae": float(np.median(np.abs(err))),
        "max_error": float(np.abs(err).max()),
        "mape": float(np.abs(err[nz] / y[nz]).mean() * 100) if nz.any() else float("nan"),
        "smape": float((2 * np.abs(err)[denom > 0] / denom[denom > 0]).mean() * 100)
        if (denom > 0).any()
        else 0.0,
        "within_10pct": float((np.abs(err[nz]) <= 0.1 * np.abs(y[nz])).mean())
        if nz.any()
        else float("nan"),
        "bias": float(err.mean()),
    }
    if naive is not None:
        naive = np.asarray(naive, float)
        ok = ~np.isnan(naive)
        base = np.abs(naive[ok] - y[ok]).mean() if ok.any() else 0.0
        out["mae_naive"] = float(base)
        out["mase"] = float(np.abs(err[ok]).mean() / base) if base else float("nan")
    return out
