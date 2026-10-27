from ml_training.clean import _belongs, main


def _tree(root):
    for p in [
        "data/raw/ds/a.jpg",
        "data/raw/ds-seg/a.jpg",
        "data/processed/ds/meta.json",
        "models/ds/yolo11n.pt",
        "models/pretrained/yolo11n.pt",
        "reports/ds/yolo11n/metrics.json",
        "reports/ds-seg/yolo11n/metrics.json",
        "configs/model/yolo11n.yaml",
    ]:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text("x")


def test_clean_one_dataset_keeps_others(tmp_path):
    _tree(tmp_path)
    assert main(["ds", "--yes", "--root", str(tmp_path)]) == 0
    assert not (tmp_path / "models/ds").exists()
    assert not (tmp_path / "reports/ds").exists()
    assert (tmp_path / "data/raw/ds/a.jpg").exists()  # raw kept without --raw
    assert (tmp_path / "reports/ds-seg").exists()
    assert (tmp_path / "models/pretrained/yolo11n.pt").exists()
    assert main(["ds", "--raw", "--yes", "--root", str(tmp_path)]) == 0
    assert not (tmp_path / "data/raw/ds").exists()
    assert (tmp_path / "data/raw/ds-seg").exists()


def test_clean_all_and_dry_run(tmp_path):
    _tree(tmp_path)
    assert main(["all", "--dry-run", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "models/ds").exists()
    assert main(["all", "--yes", "--root", str(tmp_path)]) == 0
    assert not (tmp_path / "reports/ds-seg").exists()
    assert (tmp_path / "models/pretrained/yolo11n.pt").exists()
    assert (tmp_path / "data/raw/ds/a.jpg").exists()
    assert main(["../x", "--yes", "--root", str(tmp_path)]) == 2


def test_belongs():
    models = {"yolo11n", "random_forest"}
    assert _belongs("coco8-yolo11n", "coco8", models)
    assert not _belongs("coco8-seg-yolo11n", "coco8", models)
