import numpy as np
import pandas as pd

from ml_training.xai import analysis as A
from ml_training.xai.report import write_xai_report


def test_match_image_error_types():
    gts = [(0, [0, 0, 100, 100]), (1, [200, 200, 300, 300]), (0, [500, 500, 520, 520])]
    preds = [
        {"cls": 0, "conf": 0.9, "box": [0, 0, 100, 100]},  # TP
        {"cls": 0, "conf": 0.8, "box": [2, 2, 100, 100]},  # duplicate
        {"cls": 0, "conf": 0.7, "box": [200, 200, 300, 300]},  # class
        {"cls": 1, "conf": 0.6, "box": [250, 250, 350, 350]},  # localization (IoU 0.14)
        {"cls": 0, "conf": 0.5, "box": [800, 800, 900, 900]},  # background
    ]
    p, g = A.match_image(preds, gts)
    assert [r["outcome"] for r in p] == ["TP", "FP", "FP", "FP", "FP"]
    assert [r["error"] for r in p] == ["", "duplicate", "class", "localization", "background"]
    assert [r["detected"] for r in g] == [True, False, False]
    assert g[1]["fn_type"] == "wrong class" and g[2]["fn_type"] == "missed"
    assert g[2]["size"] == "small" and g[0]["size"] == "large"


def test_wilson_and_prf():
    lo, hi = A.wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97
    p, r, f = A.prf(8, 2, 2)
    assert abs(p - 0.8) < 1e-9 and abs(r - 0.8) < 1e-9 and abs(f - 0.8) < 1e-9


def test_calibration_perfect_and_bad():
    conf = np.array([0.9] * 10 + [0.1] * 10)
    good = np.array([1] * 9 + [0] + [0] * 9 + [1])
    _, m = A.calibration(conf, good)
    assert m["ece"] < 1e-9
    _, m = A.calibration(conf, 1 - good)
    assert m["ece"] > 0.7 and m["brier"] > 0.5


def test_box_attention_and_spread():
    cam = np.zeros((100, 100))
    cam[10:20, 10:20] = 1
    a = A.box_attention(cam, [10, 10, 20, 20])
    assert a["inside"] == 1 and a["peak_in"] == 1 and abs(a["lift"] - 100) < 1e-6
    assert A.box_attention(cam, [50, 50, 60, 60])["inside"] == 0
    assert A.heat_spread(cam) < A.heat_spread(np.ones((100, 100))) == 1.0


def test_threshold_sweep_and_bootstrap():
    imgs = [
        (
            [
                {"cls": 0, "conf": 0.9, "box": [0, 0, 10, 10]},
                {"cls": 0, "conf": 0.2, "box": [50, 50, 60, 60]},
            ],
            [(0, [0, 0, 10, 10])],
        )
    ]
    s = A.threshold_sweep(imgs, ["a"], thresholds=[0.1, 0.5])
    allr = s[s["class"] == "ALL"].set_index("conf")
    assert allr.loc[0.1, "FP"] == 1 and allr.loc[0.5, "FP"] == 0 and allr.loc[0.5, "f1"] == 1
    per_image = pd.DataFrame({"TP": [5, 3, 4], "FP": [1, 0, 2], "FN": [0, 1, 1]})
    ci = A.bootstrap_prf(per_image, 200)
    assert ci["precision"][0] <= 12 / 15 <= ci["precision"][1]


def test_tests_and_report(tmp_path):
    rng = np.random.default_rng(0)
    det = pd.DataFrame({
        "cls": rng.integers(0, 2, 60),
        "conf": np.r_[rng.uniform(.6, 1, 40), rng.uniform(.2, .5, 20)],
        "outcome": ["TP"] * 40 + ["FP"] * 20, "iou": np.r_[rng.uniform(.5, 1, 40), np.zeros(20)],
        "error": [""] * 40 + ["background", "class"] * 10,
        "grad_lift": np.r_[rng.uniform(2, 5, 40), rng.uniform(0, 2, 20)],
        "grad_peak_in": rng.integers(0, 2, 60),
    })  # fmt: skip
    gt = pd.DataFrame({
        "cls": rng.integers(0, 2, 50), "detected": [True] * 40 + [False] * 10,
        "fn_type": [""] * 40 + ["missed"] * 10, "size": rng.choice(["small", "large"], 50),
        "box": [[0, 0, 10 + i, 10 + i] for i in range(50)], "eigen_lift": rng.uniform(0, 3, 50),
    })  # fmt: skip
    tests = A.hypothesis_tests(det, gt)
    first = tests.iloc[0]
    assert first["significant (p<0.05)"] == "yes" and first["effect_size"] > 0.5
    pc = A.per_class_table(det, gt, ["a", "b"], None)
    assert pc.TP.sum() == 40 and pc.FP.sum() == 20
    calib, _ = A.calibration(det.conf, det.outcome == "TP")
    path = write_xai_report(tmp_path / "x.xlsx", {"dataset": "t"}, [("f1", 0.5, "")], tests,
                            [("PerClass", pc, "")], calib, None, None, None, [])  # fmt: skip
    assert path.exists() and path.stat().st_size > 5000


def test_classify_stats():
    df = pd.DataFrame({"true": [0, 0, 1, 1, 1, 0], "pred": [0, 0, 1, 0, 1, 1],
                       "conf": [.9, .8, .9, .4, .7, .3], "grad_spread": [.2] * 6})  # fmt: skip
    df["correct"] = df.true == df.pred
    pc = A.classify_per_class(df, ["a", "b"])
    assert pc.set_index("class").loc["b", "most confused with"] == "a"
    cov = A.coverage_sweep(df.conf, df.correct).set_index("conf")
    assert cov.loc[0.0, "coverage"] == 1 and cov.loc[0.75, "accuracy"] == 1
