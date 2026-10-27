import json

from ml_training import lineage


def test_fingerprint_changes_with_content(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("1")
    (tmp_path / "b.txt").write_text("2")
    fp1 = lineage.fingerprint([a, tmp_path / "b.txt"], tmp_path)
    assert fp1["files"] == 2 and fp1["bytes"] == 2
    assert lineage.fingerprint([tmp_path / "b.txt", a], tmp_path) == fp1  # order independent
    a.write_text("3")
    assert lineage.fingerprint([a, tmp_path / "b.txt"], tmp_path)["sha256"] != fp1["sha256"]


def test_record_and_view(tmp_path, capsys):
    hist = tmp_path / "logs" / "history.jsonl"
    lineage.record("train", hist, dataset="ds", model="yolo11n", metrics={"map50_95": 0.5})
    lineage.record("predict", hist, dataset="ds", n_predictions=3)
    rows = [json.loads(x) for x in hist.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["train", "predict"] and rows[0]["time"]
    assert lineage.main(["5", "--excel", "--file", str(hist)]) == 0
    out = capsys.readouterr().out
    assert "predict" in out and (tmp_path / "logs" / "history.xlsx").exists()


def test_env_info_has_versions():
    env = lineage.env_info()
    assert env["python"] and "mlflow" in env["packages"]
