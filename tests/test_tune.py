import optuna
import pytest
from hydra import compose, initialize_config_dir

from ml_training.hpo import suggest_params
from ml_training.utils import CONFIG_DIR, PROJECT_ROOT

PRESETS = sorted(p.stem for p in (PROJECT_ROOT / "configs" / "tune").glob("*.yaml"))


@pytest.mark.parametrize("preset", PRESETS)
def test_tune_presets_compose_and_sample(preset):
    with initialize_config_dir(CONFIG_DIR, version_base="1.3"):
        cfg = compose("config", overrides=["data=coco8", "model=yolo11n", f"tune={preset}"])
    assert cfg.tune.name == preset
    space = cfg.tune.search_space or cfg.model.search_space
    study = optuna.create_study()
    params = suggest_params(study.ask(), space)
    assert params and set(params) == set(space)


def test_cli_override_beats_preset():
    with initialize_config_dir(CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            "config", overrides=["data=coco8", "model=yolo11n", "tune=accurate", "hpo.n_trials=3"]
        )
    assert cfg.hpo.n_trials == 3 and cfg.model.params.epochs == 300


def test_default_preset_leaves_tabular_models_alone():
    with initialize_config_dir(CONFIG_DIR, version_base="1.3"):
        cfg = compose("config")
    assert cfg.tune.name == "default" and "epochs" not in cfg.model.params
