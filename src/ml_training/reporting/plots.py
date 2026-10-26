"""PNG figures for MLflow / reports/figures (Excel has its own native charts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless – works on servers and Windows without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ml_training.metrics import EvalResult  # noqa: E402


def save_figures(result: EvalResult, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [_confusion(result, out_dir / "confusion_matrix.png")]
    if result.roc:
        paths.append(_roc(result, out_dir / "roc_curve.png"))
    return paths


def _confusion(result: EvalResult, path: Path) -> Path:
    cm = result.confusion.to_numpy()
    labels = [c.replace("true: ", "") for c in result.confusion.index]
    size = max(4, 0.8 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.size else 0
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _roc(result: EvalResult, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, df in result.roc.items():
        ax.plot(df.fpr, df.tpr, label=str(name))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    auc = result.summary.get("roc_auc")
    ax.set_title(f"ROC (AUC={auc:.3f})" if auc is not None else "ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
