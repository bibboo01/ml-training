"""Optuna hyperparameter search driven entirely by ``model.search_space`` in the YAML config."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import optuna
from omegaconf import DictConfig, OmegaConf

from ml_training.utils import log, sqlite_uri


def suggest_params(trial: optuna.Trial, search_space: DictConfig | dict) -> dict[str, Any]:
    space = (
        OmegaConf.to_container(search_space, resolve=True)
        if isinstance(search_space, DictConfig)
        else search_space
    )
    params: dict[str, Any] = {}
    for name, spec in (space or {}).items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif kind == "float":
            kwargs = {"log": spec.get("log", False)}
            if spec.get("step") is not None:
                kwargs["step"] = spec["step"]
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], **kwargs)
        elif kind == "categorical":
            choices = spec["choices"]
            # Optuna needs hashable primitives: encode lists (e.g. hidden_dims) as strings.
            encoded = [str(c) if isinstance(c, list | tuple) else c for c in choices]
            picked = trial.suggest_categorical(name, encoded)
            params[name] = choices[encoded.index(picked)]
        else:
            raise ValueError(f"Unknown search-space type {kind!r} for {name}")
    return params


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    return {
        "tpe": lambda: optuna.samplers.TPESampler(seed=seed),
        "random": lambda: optuna.samplers.RandomSampler(seed=seed),
        "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=seed),
    }[name]()


def make_pruner(name: str) -> optuna.pruners.BasePruner:
    return optuna.pruners.MedianPruner() if name == "median" else optuna.pruners.NopPruner()


def run_study(
    hpo_cfg: DictConfig,
    seed: int,
    objective: Callable[[optuna.Trial], float],
    resume: bool = False,
) -> optuna.Study:
    """resume=True keeps the stored study and only runs the trials that are still missing."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = sqlite_uri(hpo_cfg.storage)
    keep = bool(hpo_cfg.load_if_exists or (resume and storage))
    if storage and not keep:
        # Not resuming: drop an old study with the same name so re-runs just work.
        with contextlib.suppress(KeyError):
            optuna.delete_study(study_name=hpo_cfg.study_name, storage=storage)
    study = optuna.create_study(
        study_name=hpo_cfg.study_name,
        storage=storage,
        load_if_exists=keep,
        direction=hpo_cfg.direction,
        sampler=make_sampler(hpo_cfg.sampler, seed),
        pruner=make_pruner(hpo_cfg.pruner),
    )
    n_trials = hpo_cfg.n_trials
    if resume:
        # a trial cut off by the crash stays "RUNNING" forever – mark it failed, it is redone
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            with contextlib.suppress(Exception):
                study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
        n_trials = max(0, n_trials - finished_trials(study))
        log.info("Optuna resume: %d trials done, %d to go", finished_trials(study), n_trials)
    if n_trials:
        study.optimize(objective, n_trials=n_trials, timeout=hpo_cfg.timeout, gc_after_trial=True)
    return study


def finished_trials(study: optuna.Study) -> int:
    done = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(t.state in done for t in study.trials)


def trials_dataframe(study: optuna.Study) -> Any:
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    if "duration" in df:
        df["duration_s"] = df.pop("duration").dt.total_seconds()
    complete = df["state"] == "COMPLETE"
    best = df["value"].where(complete)
    df["best_so_far"] = (
        best.cummax() if study.direction.name == "MAXIMIZE" else best.cummin()
    ).ffill()
    df.columns = [c.replace("params_", "") for c in df.columns]
    return df


def param_importances(study: optuna.Study) -> dict[str, float]:
    try:
        return optuna.importance.get_param_importances(study)
    except Exception:
        return {}
