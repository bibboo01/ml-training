import numpy as np

from ml_training.metrics import evaluate


def test_binary_counts():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    r = evaluate(y_true, y_pred, None, ["neg", "pos"], average="binary")
    assert (r.summary["TP"], r.summary["FP"], r.summary["FN"], r.summary["TN"]) == (3, 1, 1, 3)
    assert r.summary["precision"] == 0.75
    assert r.summary["recall"] == 0.75
    assert r.summary["specificity"] == 0.75


def test_multiclass_one_vs_rest():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 2, 1, 0])
    r = evaluate(y_true, y_pred, None, ["a", "b", "c"])
    pc = r.per_class.set_index("class")
    assert pc.loc["b", "FN"] == 1 and pc.loc["c", "FP"] == 1
    assert (pc[["TP", "FP", "FN", "TN"]].sum(axis=1) == len(y_true)).all()
