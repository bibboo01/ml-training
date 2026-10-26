"""End-to-end smoke test: prepare -> featurize -> train (3 trials) on iris in a temp dir."""

from hydra import compose, initialize_config_dir

from ml_training import dataset, features
from ml_training.modeling import train
from ml_training.utils import CONFIG_DIR


def test_end_to_end(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=iris",
                "model=logistic_regression",
                "hpo.n_trials=3",
                f"paths.root={tmp_path.as_posix()}",
                "mlflow.register_model=false",
            ],
        )
    dataset.run(cfg)
    features.run(cfg)
    summary = train.train(cfg)
    assert summary["accuracy"] > 0.7
    assert (tmp_path / "reports/iris/logistic_regression/report.xlsx").exists()
