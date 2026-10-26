import optuna

from ml_training.hpo import suggest_params


def test_suggest_params_types():
    space = {
        "n": {"type": "int", "low": 1, "high": 5},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        "dims": {"type": "categorical", "choices": [[32], [64, 32]]},
    }
    study = optuna.create_study()
    params = suggest_params(study.ask(), space)
    assert 1 <= params["n"] <= 5
    assert 1e-4 <= params["lr"] <= 1e-1
    assert params["dims"] in ([32], [64, 32])
