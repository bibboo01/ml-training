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


def _labels(root, rows):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "images" / "a.jpg").write_bytes(b"x")
    (root / "labels" / "a.txt").write_text("\n".join(rows) + "\n")


@pytest.mark.parametrize(
    ("rows", "task"),
    [
        (["0 0.5 0.5 0.2 0.2"], "detect"),
        (["0 0.1 0.1 0.2 0.1 0.3 0.3 0.1 0.4 0.2 0.2"], "segment"),
        (["0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"], "obb"),
    ],
)
def test_detect_task_from_labels(tmp_path, rows, task):
    from omegaconf import OmegaConf

    from ml_training.detection_data import detect_task

    _labels(tmp_path, rows)
    assert detect_task(tmp_path, OmegaConf.create({})) == task


def test_detect_task_pose_and_classify(tmp_path):
    from omegaconf import OmegaConf

    from ml_training.detection_data import detect_task

    pose = tmp_path / "pose"
    _labels(pose, ["0 0.5 0.5 0.2 0.2 0.5 0.5 2"])
    (pose / "data.yaml").write_text("names: [p]\nkpt_shape: [1, 3]\n")
    assert detect_task(pose, OmegaConf.create({})) == "pose"
    cls = tmp_path / "cls"
    for c in ("cat", "dog"):
        (cls / "train" / c).mkdir(parents=True)
        (cls / "train" / c / "1.jpg").write_bytes(b"x")
    assert detect_task(cls, OmegaConf.create({})) == "classify"


def test_classify_featurize_splits_per_class(tmp_path):
    from omegaconf import OmegaConf

    from ml_training.detection_data import featurize
    from ml_training.utils import read_json

    raw = tmp_path / "raw" / "pets"
    for c in ("cat", "dog"):
        (raw / c).mkdir(parents=True)
        for i in range(10):
            (raw / c / f"{i}.jpg").write_bytes(b"x")
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "proc")},
            "data": {"name": "pets", "source": "yolo", "task": "auto", "val_size": 0.2},
        }
    )
    out = featurize(cfg)
    meta = read_json(out / "meta.json")
    assert meta["task"] == "classify" and meta["class_names"] == ["cat", "dog"]
    assert meta["n_rows"] == {"train": 16, "val": 4}
    assert len(list((out / "val" / "dog").iterdir())) == 2


def test_task_weights_and_classify_counts():
    pytest.importorskip("ultralytics")
    from ml_training.modeling.yolo import classify_counts, task_weights

    assert task_weights("yolo11n.pt", "segment") == "yolo11n-seg.pt"
    assert task_weights("yolo26x.pt", "classify") == "yolo26x-cls.pt"
    assert task_weights("yolov8s.pt", "pose") == "yolov8s-pose.pt"
    assert task_weights("yolo11n.pt", "detect") == "yolo11n.pt"
    assert task_weights("models/pretrained/my.pt", "segment") == "models/pretrained/my.pt"
    m = np.array([[4, 1], [0, 5]])  # [pred, true]
    df = classify_counts(m, ["a", "b"]).set_index("class")
    assert df.loc["a", ["TP", "FP", "FN", "TN"]].tolist() == [4, 1, 0, 5]
    assert df.loc["b", ["TP", "FP", "FN", "TN"]].tolist() == [5, 0, 1, 4]
