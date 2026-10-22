import json

import numpy as np
import pytest
import torch

from pretrain.pipeline import finetuning, frozen, training
from pretrain.pipeline.budget import BudgetStop, NoGpuBudget
from pretrain.pipeline.data import Windows
from tests.test.pipeline.data import prepared_fixture
from tests.test.pipeline.training import configuration


def test_finetuning_uses_declared_batches_and_resumes_during_validation(tmp_path, monkeypatch):
    manifest = prepared_fixture(tmp_path / "data")
    checkpoint = training.run(configuration(), manifest, tmp_path / "pretrain")
    config = {**configuration(), "epochs": 2}
    observed = []
    observed_rates = []
    original_batches = finetuning.batches
    original_step = torch.optim.AdamW.step

    def checked_batches(view, **kwargs):
        observed.append(kwargs["batch_size"])
        return original_batches(view, **kwargs)

    monkeypatch.setattr(finetuning, "batches", checked_batches)

    def checked_step(optimizer, *args, **kwargs):
        observed_rates.append(optimizer.param_groups[-1]["lr"])
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", checked_step)
    finetuning.run(config, manifest, checkpoint, tmp_path / "whole")
    assert observed and set(observed) == {config["logical_batch"]}

    updates = (len(Windows(manifest, "train")) + config["logical_batch"] - 1) // config[
        "logical_batch"
    ]
    assert observed_rates[0] == pytest.approx(
        finetuning.learning_rate(1 / updates, config["epochs"], warmup=config["warmup_epochs"])
    )
    receipt = json.loads((tmp_path / "whole" / "run.json").read_text())
    assert receipt["resolved"]["scheduler"]["warmup_epochs"] == config["warmup_epochs"]
    assert receipt["resolved"]["optimizer"]["defaults"]["betas"] == [0.9, 0.999]

    class StopOnValidation(NoGpuBudget):
        def __init__(self):
            self.calls = 0

        def check(self):
            self.calls += 1
            if self.calls == updates + 1:
                raise BudgetStop("interrupted validation")

    with pytest.raises(BudgetStop):
        finetuning.run(
            config, manifest, checkpoint, tmp_path / "partial", budget=StopOnValidation()
        )
    finetuning.run(
        config,
        manifest,
        checkpoint,
        tmp_path / "resumed",
        resume=tmp_path / "partial" / "last.pth",
    )
    left = torch.load(tmp_path / "whole" / "last.pth", weights_only=True)
    right = torch.load(tmp_path / "resumed" / "last.pth", weights_only=True)
    for name, value in left["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)
    assert left["state"] == right["state"]
    assert torch.equal(left["rng"]["torch_rng"], right["rng"]["torch_rng"])


def test_both_downstream_modes_select_before_opening_test(tmp_path, monkeypatch):
    manifest = prepared_fixture(tmp_path / "data")
    config = configuration()
    checkpoint = training.run(config, manifest, tmp_path / "pretrain")
    for module, name in ((frozen, "frozen"), (finetuning, "finetune")):
        output = tmp_path / name
        original = module.Windows

        def checked_view(path, split, **kwargs):
            if split == "test":
                selected = json.loads((output / "selection.json").read_text())
                assert selected["status"] == "locked_before_test"
                assert not (output / "test_predictions.npz").exists()
            return original(path, split, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(module, "Windows", checked_view)
            scores = module.run(config, manifest, checkpoint, output)
        assert 0 <= scores["balanced_accuracy"] <= 1
        assert 0 <= scores["auroc"] <= 1
        saved = np.load(output / "test_predictions.npz")
        assert len(saved["labels"]) == 8
        assert json.loads((output / "result.json").read_text())["status"] == "complete"
        backend = json.loads((output / "attention.json").read_text())
        assert backend and set(backend.values()) == {"sdpa_math"}


def test_final_test_interruption_preserves_last_finetuning_state(tmp_path, monkeypatch):
    manifest = prepared_fixture(tmp_path / "data")
    checkpoint = training.run(configuration(), manifest, tmp_path / "pretrain")
    config = {**configuration(), "epochs": 2}
    output = tmp_path / "partial"
    original = finetuning.predictions
    validation_calls = []

    def controlled_validation(model, view, *args):
        labels, probabilities, identities = original(model, view, *args)
        if view.split == "validation":
            validation_calls.append(True)
            # Force the first checkpoint to win while real training continues.
            classes = labels if len(validation_calls) == 1 else 1 - labels
            probabilities = np.eye(2)[classes]
        return labels, probabilities, identities

    monkeypatch.setattr(finetuning, "predictions", controlled_validation)

    class StopDuringTest(NoGpuBudget):
        final_training = None

        def check(self):
            if (output / "selection.json").exists():
                self.final_training = torch.load(output / "last.pth", weights_only=True)
                raise BudgetStop("interrupted final test")

    budget = StopDuringTest()
    with pytest.raises(BudgetStop, match="final test"):
        finetuning.run(config, manifest, checkpoint, output, budget=budget)
    before = budget.final_training
    after = torch.load(output / "last.pth", weights_only=True)
    assert torch.load(output / "best.pth", weights_only=True)["epoch"] == 1
    for name, value in before["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, after["model"][name], rtol=0, atol=0)
    assert before["state"] == after["state"]
    assert torch.equal(before["rng"]["torch_rng"], after["rng"]["torch_rng"])
    resumed = tmp_path / "resumed"
    finetuning.run(config, manifest, checkpoint, resumed, resume=output / "last.pth")
    assert json.loads((resumed / "result.json").read_text())["status"] == "complete"
