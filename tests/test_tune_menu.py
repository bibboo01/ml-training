import yaml

from ml_training import tune_menu


def test_create_preset_from_answers():
    answers = iter(["4", "150", "800", "-1", "", "10", "adamw", "0", "5", "", "zz_test_preset"])
    path = tune_menu.TUNE_DIR / "zz_test_preset.yaml"
    try:
        assert tune_menu.create(read=lambda _prompt: next(answers)) == 0
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        p = data["model"]["params"]
        assert (p["epochs"], p["imgsz"], p["batch"], p["freeze"]) == (150, 800, -1, 10)
        assert p["optimizer"] == "AdamW" and p["device"] == 0
        assert data["hpo"]["n_trials"] == 5
        assert "optimizer" in data["tune"]["search_space"]  # copied from 'accurate'
        assert "zz_test_preset" in tune_menu.presets()
        assert tune_menu.check("zz_test_preset")[0]
    finally:
        path.unlink(missing_ok=True)


def test_parse_values():
    assert tune_menu._parse("device", "auto") is None
    assert tune_menu._parse("device", "0,1") == "0,1"
    assert tune_menu._parse("device", "CPU") == "cpu"
    assert tune_menu._parse("optimizer", "sgd") == "SGD"
