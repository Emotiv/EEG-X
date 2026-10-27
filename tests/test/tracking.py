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
        self.config = {}
        self.artifacts = []
        self.rows = []
        self.metrics = []

    def log(self, values):
        self.rows.append(values)

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def finish(self, **kwargs):
        self.finished = kwargs

    def log_artifact(self, artifact, **kwargs):
        self.artifacts.append((artifact, kwargs))


def test_neuralbench_root_metrics_keep_real_optimizer_and_step_values():
    from pretrain.tracking import Tracker

    run = RecordedRun()
    logger = Tracker(run, optimizer_name="StableAdamW")
    logger.log({"epoch": 1.5, "trainer/global_step": 3, "train/lr": 0.0002})
    logger.log({"epoch": 2, "val/raw_mse": 0.4})
    logger.summary({"best_epoch": 1})
    assert "lr-AdamW" not in run.rows[0]
    assert run.rows[0]["lr-StableAdamW"] == 0.0002
    assert run.rows[0]["trainer/global_step"] == 3
    assert run.rows[1]["val/loss"] == run.rows[1]["val/raw_mse"]
    assert run.rows[-1] == {"epoch": 2, "trainer/global_step": 3, "best_epoch": 1}


def test_resume_uses_same_run_and_seed_recipes_stay_stable_without_preset_groups(
    tmp_path, monkeypatch
):
    calls, runs = [], []

    def initialize(**kwargs):
        calls.append(kwargs)
        run = RecordedRun()
        runs.append(run)
        return run

    monkeypatch.setattr(wandb, "init", initialize)
    metadata = {"task": "dreamer_arousal", "fingerprint": "frozen-source"}
    settings = {
        "enabled": True,
        "project": "test-project",
        "name": "test_pretrain_seed_{seed}",
    }
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
    assert calls[0]["config"]["domain"] == "in_domain"
    assert calls[0]["config"]["evaluation_mode"] == "pretrain"
    assert calls[0]["settings"].console == "wrap"
    assert calls[0]["settings"].x_disable_stats is False
    assert calls[0]["name"] == "test_pretrain_seed_241"
    assert runs[0].summary["run_status"] == "complete"
    assert json.loads((resumed / "wandb.json").read_text())["id"] == "saved-run-id"
    second_seed = tmp_path / "second-seed"
    second_seed.mkdir()
    config = resolve_training_config({"seed": 242, "input_samples": 256})
    with track(settings, config, metadata, second_seed, stage="pretrain"):
        pass
    assert calls[0]["group"] is None
    assert calls[2]["group"] is None
    assert calls[0]["config"]["recipe_id"] == calls[2]["config"]["recipe_id"]
    assert calls[2]["config"]["seed"] == 242
    assert calls[2]["name"] == "test_pretrain_seed_242"


def test_embedded_probes_share_pretraining_run_then_finetuning_logs_final_test(
    tmp_path, monkeypatch
):
    import torch
    import yaml

    from pretrain.train import run_config
    from tests.test.pipeline.data import prepared_fixture

    prepared_fixture(tmp_path / "data")
    calls, runs = [], []

    def initialize(**kwargs):
        calls.append(kwargs)
        run = RecordedRun()
        runs.append(run)
        return run

    monkeypatch.setattr(wandb, "init", initialize)
    monkeypatch.setattr(wandb.plot, "confusion_matrix", lambda **kwargs: "confusion-table")
    settings = {
        "dataset": "data.h5",
        "purpose": "engineering",
        "output": "pretrain",
        "device": "cpu",
        "precision": "fp32",
        "seed": 241,
        "epochs": 2,
        "warmup_epochs": 0,
        "depth": 1,
        "width": 16,
        "logical_batch": 28,
        "microbatch": 14,
        "workers": 0,
        "probe_interval": 1,
        "diagnostic_test_probes": True,
        "objective": {"name": "raw_soft_dict_v2", "weight": 1.0,
                      "temperature": 1.0, "k": 2, "groups": 2,
                      "kernel_sizes": [3], "dilations": [1]},
        "wandb": {"enabled": True, "name": "test_embedded_seed_{seed}"},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(settings))
    run_config(path)
    assert len(calls) == 1
    pretraining = runs[0]
    for prefix in ("linear_probe/val", "linear_probe/test"):
        rows = [row for row in pretraining.rows if f"{prefix}/acc" in row]
        assert [row["epoch"] for row in rows] == [1, 2]
        assert f"selected/{prefix}/acc" in pretraining.summary
        metrics = {"loss", "acc", "bal_acc", "auroc", "auprc", "f1_score_micro", "f1_score_macro",
                   "weighted_f1", "precision_macro", "recall_macro", "f1_score_binary"}
        assert all({prefix + "/" + key for key in metrics} <= set(row) for row in rows)
    validation_rows = [row for row in pretraining.rows if "val/soft_dict" in row]
    assert [row["epoch"] for row in validation_rows] == [1, 2]
    assert all(row["val/soft_dict"] >= 0 for row in validation_rows)
    assert all(row["val/loss"] == row["val/raw_mse"] for row in validation_rows)
    assert any("train/loss_step" in row for row in pretraining.rows)
    updates = [row for row in pretraining.rows if "train/loss_step" in row]
    assert [row["trainer/global_step"] for row in updates] == list(range(1, len(updates) + 1))
    assert all(row["lr-StableAdamW"] == row["train/lr"] for row in updates)
    assert any("best_epoch" in row for row in pretraining.rows)
    assert [row["epoch"] for row in pretraining.rows if "train/loss_epoch" in row] == [1, 2]
    assert "selected/linear_probe/val/bal_acc" in pretraining.summary
    assert not any("balanced_accuracy" in key for row in pretraining.rows for key in row)
    assert not any(key.startswith("test/") for row in pretraining.rows for key in row)
    checkpoint = tmp_path / "pretrain/training/best.pth"
    saved = torch.load(tmp_path / "pretrain/training/last.pth", weights_only=True)
    assert saved["state"]["best_loss"] == pretraining.summary["selected/validation_mse"]
    assert saved["state"]["best_epoch"] == pretraining.summary["selected/epoch"]
    for name in ("1", "2", "selected"):
        head = torch.load(tmp_path / f"pretrain/training/probes/{name}/best.pth", weights_only=True)
        expected = max(head["history"], key=lambda row: row["validation"]["balanced_accuracy"])
        assert head["epoch"] == expected["epoch"]
    assert pretraining.summary["best_epoch"] == pretraining.summary["selected/epoch"]
    assert pretraining.config["n_total_params"] > 0
    assert pretraining.config["n_trainable_params"] == pretraining.config["n_total_params"]
    assert pretraining.artifacts[0][1]["aliases"] == ["best"]
    assert "best.pth" in pretraining.artifacts[0][0].manifest.entries
    assert any("linear_probe/val/confusion_matrix" in row for row in pretraining.rows)
    assert any("linear_probe/test/confusion_matrix" in row for row in pretraining.rows)
    subjects = json.loads((tmp_path / "pretrain/split.json").read_text())["subjects"]
    groups = [
        {s for s, role in subjects.items() if role == split}
        for split in ("train", "validation", "test")
    ]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]

    settings.update(
        stage="finetune", checkpoint=str(checkpoint), output="finetune", probe_interval=0
    )
    settings["wandb"]["name"] = "test_finetune_seed_{seed}"
    path.write_text(yaml.safe_dump(settings))
    run_config(path)
    assert len(calls) == 2
    assert calls[0]["config"]["selection_metric"] == "val/raw_mse"
    assert calls[1]["config"]["selection_metric"] == "val/bal_acc"
    assert calls[1]["config"]["selection_mode"] == "max"
    assert calls[1]["config"]["evaluation_mode"] == "finetune"
    tuning = runs[1]
    assert [row["epoch"] for row in tuning.rows if "val/acc" in row] == [1, 2]
    assert [row["epoch"] for row in tuning.rows if "val/confusion_matrix" in row] == [1, 2]
    assert all(row["val/loss"] >= 0 for row in tuning.rows if "val/acc" in row)
    assert any(
        "train/lr" in row and row["train/lr_min"] <= row["train/lr_max"] for row in tuning.rows
    )
    assert any("train/loss_step" in row for row in tuning.rows)
    updates = [row for row in tuning.rows if "train/loss_step" in row]
    assert [row["trainer/global_step"] for row in updates] == list(range(1, len(updates) + 1))
    assert all(row["lr-AdamW"] == row["train/lr"] for row in updates)
    assert any("best_epoch" in row for row in tuning.rows)
    assert [row["epoch"] for row in tuning.rows if "train/loss_epoch" in row] == [1, 2]
    assert any("val/bal_acc" in row for row in tuning.rows)
    assert tuning.config["n_total_params"] > 0
    assert tuning.artifacts[0][1]["aliases"] == ["best"]
    assert "best.pth" in tuning.artifacts[0][0].manifest.entries
    final = [row for row in tuning.rows if "test/acc" in row]
    assert len(final) == 1
    assert final[0]["epoch"] == 2
    assert final[0]["test/acc"] == tuning.summary["test/acc"]
    assert final[0]["test/loss"] == tuning.summary["test/loss"]
    matrix = [row["test/confusion_matrix"] for row in tuning.rows if "test/confusion_matrix" in row]
    assert len(matrix) == 1 and isinstance(matrix[0], wandb.Image)
    result = json.loads((tmp_path / "finetune/training/result.json").read_text())
    assert result["selection"]["status"] == "locked_before_test"
    assert result["selection"]["criterion"] == {"metric": "val/bal_acc", "mode": "max"}
    assert tuning.summary["selected/epoch"] == result["selection"]["epoch"]
    assert tuning.summary["test/acc"] == result["test"]["acc"]
    assert tuning.summary["test/loss"] == result["test"]["loss"]
    assert tuning.summary["test/bal_acc"] == result["test"]["balanced_accuracy"]
    assert "test/balanced_accuracy" not in tuning.summary
    assert pretraining.summary["run_status"] == tuning.summary["run_status"] == "complete"

    settings.update(stage="linear_probe", output="standalone_probe")
    settings["wandb"]["name"] = "test_standalone_probe_seed_{seed}"
    path.write_text(yaml.safe_dump(settings))
    run_config(path)
    standalone = runs[2]
    events = [row for row in standalone.rows if "test/loss" in row]
    assert len(events) == 1
    assert events[0]["test/loss"] == standalone.summary["test/loss"]
    assert events[0]["test/bal_acc"] == standalone.summary["test/bal_acc"]
    assert events[0]["epoch"] == 2


def test_linear_probe_run_uses_the_same_stage_and_job_type(tmp_path, monkeypatch):
    calls = []

    def initialize(**kwargs):
        calls.append(kwargs)
        return RecordedRun()

    monkeypatch.setattr(wandb, "init", initialize)
    with track(
        {"enabled": True, "domain": "cross_domain"},
        {"seed": 241},
        {"task": "crowdsourced", "fingerprint": "target-release"},
        tmp_path,
        stage="linear_probe",
    ):
        pass
    assert calls[0]["job_type"] == "linear_probe"
    assert calls[0]["config"]["domain"] == "cross_domain"
    assert calls[0]["config"]["stage"] == "linear_probe"
    assert calls[0]["config"]["evaluation_mode"] == "linear_probe"
    assert calls[0]["name"] == "crowdsourced-linear_probe-seed-241"


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
        "dataset": "data.h5",
        "purpose": "engineering",
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


def test_linear_readout_logs_neuralbench_names_and_weighted_epoch_loss(tmp_path, monkeypatch):
    import math
    import numpy as np
    from pretrain.pipeline.budget import NoGpuBudget
    from pretrain.pipeline.readout import fit_readout
    from pretrain.tracking import Tracker

    monkeypatch.setattr(wandb.plot, "confusion_matrix", lambda **kwargs: "confusion-table")
    run = RecordedRun()
    generator = np.random.default_rng(9)
    x = generator.normal(size=(10, 4)).astype(np.float32)
    y = np.arange(10) % 2
    probe = fit_readout(
        x,
        y,
        x[:4],
        y[:4],
        {"seed": 241},
        tmp_path,
        NoGpuBudget(),
        tracker=Tracker(run),
        settings={"epochs": 3, "batch_size": 3},
    )
    steps = [row for row in run.rows if "train/loss_step" in row]
    epochs = [row for row in run.rows if "train/loss_epoch" in row]
    assert len(steps) == 12
    assert len(epochs) == 3
    for index, row in enumerate(epochs):
        batch_rows = steps[index * 4 : (index + 1) * 4]
        weighted_loss = (
            sum(value["train/loss_step"] * size for value, size in zip(batch_rows, [3, 3, 3, 1]))
            / 10
        )
        assert math.isclose(row["train/loss_epoch"], weighted_loss, rel_tol=1e-6)
        assert row["train/loss_epoch"] == probe.history[index]["train_loss"]
        assert "val/bal_acc" in row
        assert "val/balanced_accuracy" not in row


def test_final_test_scores_log_once_and_keep_summary_values():
    from pretrain.tracking import Tracker

    run = RecordedRun()
    logger = Tracker(run)
    logger.log({"epoch": 2, "trainer/global_step": 12})
    logger.final_test({"loss": 0.6, "balanced_accuracy": 0.7, "acc": 0.75})
    expected = {"test/loss": 0.6, "test/bal_acc": 0.7, "test/acc": 0.75}
    assert all(run.summary[key] == value for key, value in expected.items())
    assert run.rows[-1] == {"epoch": 2, "trainer/global_step": 12, **expected}
    with pytest.raises(RuntimeError, match="already been logged"):
        logger.final_test({"loss": 0.6})
    assert len([row for row in run.rows if "test/loss" in row]) == 1
