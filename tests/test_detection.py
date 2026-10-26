import numpy as np
import pytest

from ml_training.detection_data import _resolve_entry, img2label


def test_detection_counts_from_ultralytics_matrix():
    pytest.importorskip("ultralytics")
    from ml_training.modeling.yolo import detection_counts

    # rows = predicted, cols = true, last row/col = background
    m = np.array(
        [
            [5, 1, 2],  # pred cat: 5 correct, 1 was a dog, 2 on background
            [0, 3, 1],  # pred dog: 3 correct, 1 on background
            [2, 1, 0],  # missed: 2 cats, 1 dog
        ]
    )
    df = detection_counts(m, ["cat", "dog"]).set_index("class")
    assert df.loc["cat", ["TP", "FP", "FN"]].tolist() == [5, 3, 2]
    assert df.loc["dog", ["TP", "FP", "FN"]].tolist() == [3, 1, 2]


def test_roboflow_relative_paths(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "images" / "a.jpg").write_bytes(b"x")
    found = _resolve_entry("../train/images", tmp_path, tmp_path)
    assert [p.name for p in found] == ["a.jpg"]


def test_img2label():
    p = img2label(__import__("pathlib").Path("/d/train/images/x.jpg"))
    assert p.as_posix() == "/d/train/labels/x.txt"
