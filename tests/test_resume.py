import contextlib
import json

import optuna
import torch
from omegaconf import OmegaConf

from ml_training import resume
from ml_training.hpo import run_study
from ml_training.modeling.yolo import unfinished_checkpoint
from ml_training.runstate import write_state


def test_unfinished_checkpoint(tmp_path):
    w = tmp_path / "final" / "weights"
    w.mkdir(parents=True)
    assert unfinished_checkpoint(tmp_path / "final") is None
    torch.save({"epoch": 4, "optimizer": {"state": {}}}, w / "last.pt")
    assert unfinished_checkpoint(tmp_path / "final") == w / "last.pt"
    torch.save({"epoch": -1, "optimizer": None}, w / "last.pt")  # finished = stripped
    assert unfinished_checkpoint(tmp_path / "final") is None
    (w / "last.pt").write_bytes(b"half written")
    assert unfinished_checkpoint(tmp_path / "final") is None


def test_state_list_and_command(tmp_path):
    a, b = tmp_path / "helmet" / "yolo11s", tmp_path / "helmet" / "yolo11n"
    overrides = ["data=yolo", "data.name=helmet", "model=yolo11s", "tune=fast"]
    write_state(a, status="running", stage="hpo", overrides=overrides)
    write_state(b, status="running", stage="final")
    write_state(b, status="finished")
    assert json.loads((b / "run_state.json").read_text())["stage"] == "final"  # merged
    todo = resume.unfinished(tmp_path)
    assert [s["model"] for s in todo] == ["yolo11s"]
    assert "Optuna" in resume.describe(todo[0])
    cmd = resume.command(todo[0])
    assert cmd[1:4] == ["-m", "ml_training.modeling.train", "data=yolo"]
    assert cmd[-2:] == ["tune=fast", "resume=true"]


def test_run_study_resume_runs_only_missing_trials(tmp_path):
    cfg = OmegaConf.create({"storage": f"sqlite:///{tmp_path}/o.db", "study_name": "s",
                            "load_if_exists": False, "direction": "maximize", "sampler": "random",
                            "pruner": "none", "n_trials": 5, "timeout": None})  # fmt: skip
    calls = []

    def obj(trial):
        calls.append(trial.number)
        if len(calls) == 3:
            raise KeyboardInterrupt  # "power cut" during the 3rd trial
        return trial.suggest_float("x", 0, 1)

    with contextlib.suppress(KeyboardInterrupt):
        run_study(cfg, 0, obj)
    calls.clear()
    study = run_study(cfg, 0, lambda t: t.suggest_float("x", 0, 1) or 0.0, resume=True)
    states = [t.state for t in study.trials]
    assert states.count(optuna.trial.TrialState.COMPLETE) == 5
    assert optuna.trial.TrialState.RUNNING not in states
    study = run_study(cfg, 0, lambda t: 1.0)  # a normal run starts over
    assert len(study.trials) == 5
