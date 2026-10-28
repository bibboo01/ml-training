import json

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from ml_training import vision_data as V
from ml_training.backends import data_backend, model_backend
from ml_training.modeling import vision as M


def _cfg(tmp_path, model=None, **data):
    return OmegaConf.create(
        {
            "seed": 0,
            "paths": {
                "root": str(tmp_path),
                "raw": str(tmp_path / "raw"),
                "models": str(tmp_path / "models"),
                "processed": str(tmp_path / "processed" / "ds"),
            },
            "data": {
                "name": "ds",
                "kind": "vision",
                "path": None,
                "task": "auto",
                "class_names": None,
                "val_size": 0.2,
                "test_size": 0.2,
                **data,
            },
            "model": model
            or {
                "name": "img_tiny",
                "backend": "vision",
                "arch": "simple_cnn",
                "width": 8,
                "pretrained": False,
            },
        }
    )


def _img(path, size=32, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), value, np.uint8)).save(path)


def test_backend_dispatch(tmp_path):
    cfg = _cfg(tmp_path)
    assert data_backend(cfg).__name__.endswith("backends.vision")
    assert model_backend(cfg).__name__.endswith("backends.vision")


def test_classify_layouts(tmp_path):
    root = tmp_path / "raw" / "ds"
    for c in ("cat", "dog"):
        for i in range(10):
            _img(root / c / f"{i}.png")
    _img(root / "test" / "cat" / "t.png")  # explicit split folder: label = folder below the split
    recs = V.classify_records(root)
    assert V.detect_task(root) == "classify" and len(recs) == 21
    t = next(r for r in recs if r["image"].endswith("t.png"))
    assert (t["label"], t["split"]) == ("cat", "test")
    out = V.featurize(_cfg(tmp_path))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == "classify" and meta["class_names"] == ["cat", "dog"]
    assert sum(meta["n_rows"].values()) == 21


def test_segment_layouts_and_classes(tmp_path):
    root = tmp_path / "raw" / "ds"
    (root / "masks").mkdir(parents=True)
    m = np.zeros((32, 32), np.uint8)
    m[4:12, 4:12] = 2
    for i in range(6):
        _img(root / "images" / f"{i}.jpg")
        Image.fromarray(m).save(root / "masks" / f"{i}.png")
    assert V.detect_task(root) == "segment"
    recs = V.segment_records(root)
    assert len(recs) == 6 and recs[0]["mask"].endswith("0.png")
    assert V.segment_classes(root, recs, None) == ["background", "class_1", "class_2"]
    (root / "classes.txt").write_text("bg\nroad\ncar\n", encoding="utf-8")
    assert V.segment_classes(root, recs, None) == ["bg", "road", "car"]
    # Roboflow export: x.jpg + x_mask.png + _classes.csv
    rf = tmp_path / "rf"
    _img(rf / "train" / "a.jpg")
    Image.fromarray(np.zeros((32, 32), np.uint8)).save(rf / "train" / "a_mask.png")
    (rf / "train" / "_classes.csv").write_text(
        "Pixel Value, Class\n0, background\n1, crack\n", encoding="utf-8"
    )
    r = V.segment_records(rf)
    assert len(r) == 1 and r[0]["split"] == "train"
    assert V.segment_classes(rf, r, None) == ["background", "crack"]


def test_seg_eval_iou():
    gold = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    r = M.seg_eval(M.seg_confusion(pred, gold, 3), ["bg", "a", "b"])
    pc = r.per_class.set_index("class")
    assert pc.loc["a", "iou"] == pytest.approx(2 / 3) and pc.loc["b", "iou"] == pytest.approx(1 / 2)
    assert r.summary["pixel_accuracy"] == pytest.approx(4 / 6)
    assert r.summary["miou"] == pytest.approx((1 / 3 + 2 / 3 + 1 / 2) / 3)


@pytest.mark.parametrize("task", ["classify", "segment"])
def test_fit_save_load_predict(tmp_path, task):
    names = ["bg", "x"]
    recs = []
    for i in range(8):
        p = tmp_path / f"{i}.png"
        _img(p, value=200 * (i % 2))
        rec = {"image": str(p), "label": names[i % 2]}
        if task == "segment":
            m = np.full((32, 32), i % 2, np.uint8)
            Image.fromarray(m).save(tmp_path / f"{i}_m.png")
            rec = {"image": str(p), "mask": str(tmp_path / f"{i}_m.png")}
        recs.append(rec)
    model_cfg = {
        "name": "t",
        "backend": "vision",
        "arch": "simple_unet" if task == "segment" else "simple_cnn",
        "width": 8,
        "pretrained": False,
    }
    cfg = _cfg(tmp_path, model=model_cfg)
    p = M._params(cfg, {"epochs": 2, "batch_size": 4, "imgsz": 32, "patience": 0})
    out = tmp_path / "final"
    out.mkdir()
    model, hist, best = M.fit(cfg, task, names, recs, recs, p, out)
    assert len(hist) == 2 and (out / "last.pt").exists() and (out / "best.pt").exists()
    M.save_trained(model, cfg, task, names, p, tmp_path / "model")
    loaded, meta = M.load_trained(tmp_path / "model")
    res = M.predict_images(loaded, meta, [tmp_path / "0.png", tmp_path / "1.png"])
    if task == "segment":
        assert res[0].shape == (32, 32)
    else:
        assert res.shape == (2, 2) and np.allclose(res.sum(1), 1, atol=1e-5)
    # resume: one more epoch on top of the saved checkpoint
    _, hist2, _ = M.fit(cfg, task, names, recs, recs, {**p, "epochs": 3}, out, resume=True)
    assert len(hist2) == 3


def test_wrong_model_family(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="seg_"):
        M.build_model(cfg, "segment", ["a", "b"])
