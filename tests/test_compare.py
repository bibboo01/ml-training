import json

from ml_training.compare import collect, main


def _write(reports, ds, model, **test):
    d = reports / ds / model
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"test": test}), encoding="utf-8")


def test_compare_ranks_by_map(tmp_path, capsys):
    _write(tmp_path, "ds", "yolo11n", map50=0.5, map50_95=0.3, TP=3, FP=1, FN=2)
    _write(tmp_path, "ds", "yolo11m", map50=0.7, map50_95=0.4, TP=4, FP=1, FN=1)
    df = collect("ds", tmp_path)
    assert list(df["model"]) == ["yolo11m", "yolo11n"]
    assert main(["ds", "--reports", str(tmp_path)]) == 0
    assert (tmp_path / "ds" / "compare.xlsx").exists()
    assert main(["ds", "--best", "--reports", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip().endswith("yolo11m")
    assert main(["missing", "--reports", str(tmp_path)]) == 1
