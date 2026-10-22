"""Check seed grouping and W&B identity without sending network requests."""

import json

import pytest
import wandb

from pretrain.pipeline.training import resolve_training_config
from pretrain.tracking import track


class RecordedRun:
    id = "saved-run-id"

    def __init__(self):
        self.summary = {}
        self.rows = []
        self.metrics = []

    def log(self, values):
        self.rows.append(values)

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def finish(self, **kwargs):
        self.finished = kwargs


def test_resume_uses_same_wandb_run_and_seed_recipe_groups_are_stable(tmp_path, monkeypatch):
    calls, runs = [], []

    def initialize(**kwargs):
        calls.append(kwargs)
        run = RecordedRun()
        runs.append(run)
        return run

    monkeypatch.setattr(wandb, "init", initialize)
    metadata = {"task": "dreamer_arousal", "fingerprint": "frozen-source"}
    settings = {"enabled": True, "project": "test-project"}
    config = resolve_training_config({"seed": 241, "input_samples": 256})
    first = tmp_path / "first"
    first.mkdir()
    with track(settings, config, metadata, first, stage="pretrain") as logger:
        logger.log({"epoch": 1, "train/loss": 0.5})
    checkpoint = first / "training" / "last.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    with track(settings, config, metadata, resumed, stage="pretrain", resume=checkpoint):
        pass
    assert calls[1]["id"] == "saved-run-id"
    assert calls[1]["resume"] == "must"
    assert calls[0]["config"]["model_name"] == "EEG-X"
    assert calls[0]["config"]["task_name"] == "dreamer_arousal"
    assert runs[0].summary["run_status"] == "complete"
    assert json.loads((resumed / "wandb.json").read_text())["id"] == "saved-run-id"
    second_seed = tmp_path / "second-seed"
    second_seed.mkdir()
    config = resolve_training_config({"seed": 242, "input_samples": 256})
    with track(settings, config, metadata, second_seed, stage="pretrain"):
        pass
    assert calls[0]["group"] == calls[2]["group"]
    assert calls[2]["config"]["seed"] == 242


def test_stopped_run_is_not_marked_complete(tmp_path, monkeypatch):
    from pretrain.pipeline.budget import BudgetStop

    run = RecordedRun()
    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    with pytest.raises(BudgetStop):
        with track(
            {"enabled": True},
            {"seed": 241},
            {"task": "fixture", "fingerprint": "source"},
            tmp_path,
            stage="pretrain",
        ):
            raise BudgetStop("safe stop")
    assert run.summary["run_status"] == "stopped"
    assert run.finished["exit_code"] == 1


def test_yaml_resume_after_an_optimizer_update_keeps_weights_and_wandb_identity(
    tmp_path, monkeypatch
):
    import torch
    import yaml

    from pretrain.pipeline import training
    from pretrain.pipeline.budget import BudgetStop, NoGpuBudget
    from pretrain.train import run_config
    from tests.test.pipeline.data import prepared_fixture

    prepared_fixture(tmp_path / "data")
    calls = []

    def initialize(**kwargs):
        calls.append(kwargs)
        run = RecordedRun()
        run.id = kwargs.get("id") or f"run-{len(calls)}"
        return run

    monkeypatch.setattr(wandb, "init", initialize)
    settings = {
        "manifest": "data/manifest.json",
        "output": "whole",
        "device": "cpu",
        "precision": "fp32",
        "epochs": 1,
        "warmup_epochs": 0,
        "depth": 2,
        "width": 32,
        "logical_batch": 12,
        "microbatch": 2,
        "workers": 0,
        "probe_interval": 0,
        "objective": None,
        "optimizer": {"weight_decay": 0.0},
        "wandb": {"enabled": True, "project": "test-project"},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(settings))
    run_config(path)
    original = training.run

    class StopAfterUpdate(NoGpuBudget):
        calls = 0

        def check(self):
            self.calls += 1
            if self.calls > 1:
                raise BudgetStop("stop after first update")

    def interrupted(*args, **kwargs):
        kwargs["budget"] = StopAfterUpdate()
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "run", interrupted)
    settings["output"] = "partial"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(BudgetStop):
        run_config(path)
    monkeypatch.setattr(training, "run", original)
    settings.update(output="resumed", resume="partial/training/last.pth")
    path.write_text(yaml.safe_dump(settings))
    run_config(path)
    left = torch.load(tmp_path / "whole/training/last.pth", weights_only=True)
    right = torch.load(tmp_path / "resumed/training/last.pth", weights_only=True)
    assert left["state"] == right["state"]
    for name, value in left["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)
    assert calls[2]["id"] == "run-2"
    assert calls[2]["resume"] == "must"
