"""Automatic data check (inspect_data) and the Thai error hints (friendly)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml_training import friendly
from ml_training.inspect_data import inspect


def _img(p, value=0):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, np.uint8)).save(p)


def test_image_folders(tmp_path):
    for c in ("cat", "dog"):
        for i in range(3):
            _img(tmp_path / "pets" / c / f"{i}.jpg")
    info = inspect(tmp_path / "pets")
    assert (info["kind"], info["task"], info["n_classes"]) == ("vision", "classify", 2)
    yolo = tmp_path / "boxes"
    _img(yolo / "images" / "train" / "a.jpg")
    (yolo / "labels" / "train").mkdir(parents=True)
    (yolo / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert inspect(yolo)["kind"] == "yolo" and inspect(yolo)["task"] == "detect"
    _img(tmp_path / "raw_photos" / "x.jpg")
    with pytest.raises(ValueError, match="ยังไม่มีป้ายกำกับ"):
        inspect(tmp_path / "raw_photos")


def test_tables_and_texts(tmp_path):
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "customer_id": range(60),
            "age": rng.integers(18, 80, 60),
            "price": rng.normal(1000, 200, 60),
        }
    ).to_csv(tmp_path / "a.csv", index=False)
    info = inspect(tmp_path / "a.csv")
    assert (info["kind"], info["task"], info["target"]) == ("table", "regress", "price")
    days = pd.date_range("2024-01-01", periods=40).astype(str)
    pd.DataFrame(
        {"date": list(days) * 2, "shop": ["a"] * 40 + ["b"] * 40, "sales": rng.normal(100, 10, 80)}
    ).to_csv(tmp_path / "s.csv", index=False)
    info = inspect(tmp_path / "s.csv")
    assert (
        info["task"] == "forecast"
        and info["time_column"] == "date"
        and info["group_column"] == "shop"
    )
    pd.DataFrame(
        {
            "review": ["สินค้าดีมาก ส่งไว ประทับใจ"] * 20 + ["แย่มาก ของพัง ไม่ตรงปก"] * 20,
            "label": ["pos"] * 20 + ["neg"] * 20,
        }
    ).to_csv(tmp_path / "r.csv", index=False)
    info = inspect(tmp_path / "r.csv")
    assert (info["kind"], info["task"], info["thai"]) == ("text", "classify", True)


def test_friendly_hints():
    assert "Excel" in friendly.explain(PermissionError("report.xlsx"))
    assert "setup-gpu" in friendly.explain(ModuleNotFoundError("x", name="lightgbm"))
    assert "VRAM" in friendly.explain(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert friendly.explain(ValueError("something else")) == ""
