import json

import pytest
import torch

from pretrain.pipeline.budget import BudgetStop, NoGpuBudget
from pretrain.pipeline.downstream import classifier, optimizer_groups
from pretrain.pipeline.training import run
from pretrain.objectives import DEFAULT_HYDRA
from tests.test.pipeline.data import prepared_fixture


def configuration():
    return {
        "size": "d8w64",
        "input_samples": 256,
        "epochs": 1,
        "warmup_epochs": 0,
        "precision": "fp32",
        "seed": 241,
        "logical_batch": 12,
        "microbatch": 2,
        "workers": 0,
        "purpose": "pipeline_test",
        "objective": {**DEFAULT_HYDRA, "groups": 2, "k": 4},
    }


class StopAfterUpdate(NoGpuBudget):
    def __init__(self):
        self.calls = 0

    def check(self):
        self.calls += 1
        if self.calls > 1:
            raise BudgetStop("test stop after the first full update")


def test_real_training_resume_preserves_the_logical_update(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    config = configuration()
    run(config, manifest, tmp_path / "whole")
    with pytest.raises(BudgetStop):
        run(config, manifest, tmp_path / "partial", budget=StopAfterUpdate())
    run(config, manifest, tmp_path / "resumed", resume=tmp_path / "partial" / "last.pth")
    left = torch.load(tmp_path / "whole" / "last.pth", weights_only=True)
    right = torch.load(tmp_path / "resumed" / "last.pth", weights_only=True)
    for name, value in left["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)
    assert left["state"] == right["state"]
    assert json.loads((tmp_path / "resumed" / "result.json").read_text())["status"] == "complete"
    receipt = json.loads((tmp_path / "whole" / "run.json").read_text())
    optimizer = receipt["resolved"]["optimizer"]
    assert optimizer["class"].endswith(".StableAdamW")
    assert optimizer["defaults"]["lr"] == 2.4e-4
    assert optimizer["defaults"]["eps"] == 1e-9
    assert receipt["resolved"]["model"] == left["model"]["_extra_state"]


def test_optimizer_groups_cover_rms_encoder_and_head(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    checkpoint = run(configuration(), manifest, tmp_path / "pretrain")
    model, _ = classifier(checkpoint, 2, 241)
    groups, audit = optimizer_groups(model)
    routed = [p for group in groups for p in group["params"]]
    assert len(routed) == len({id(p) for p in model.parameters()})
    first = next(row for row in audit if "encoder.raw_projection.weight" in row["names"])
    head = next(row for row in audit if "head.2.weight" in row["names"])
    assert first["lr_scale"] < head["lr_scale"] == 1


def test_early_stopping_retains_the_lowest_validation_checkpoint(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    config = {**configuration(), "epochs": 5, "early_stopping": {"patience": 1, "min_delta": 1e6}}
    checkpoint = run(config, manifest, tmp_path / "run")
    result = json.loads((checkpoint.parent / "result.json").read_text())
    assert result["stop_reason"] == "early_stopping"
    assert result["state"]["epoch"] == 2
    assert result["state"]["best_loss"] == min(
        row["validation_mse"] for row in result["state"]["history"]
    )
    backends = json.loads((checkpoint.parent / "attention.json").read_text())
    assert backends and set(backends.values()) == {"sdpa_math"}


def test_warmup_never_restarts_the_learning_rate():
    from pretrain.pipeline.training import schedule

    assert schedule(9, 0, 1, 100, warmup_epochs=10) == pytest.approx(2.4e-4)
    assert 2.3e-4 < schedule(10, 0, 1, 100, warmup_epochs=10) < 2.4e-4
    assert schedule(99, 0, 1, 100, warmup_epochs=10) == pytest.approx(1e-5)


@pytest.mark.parametrize("phase", ["1", "selected"])
def test_resume_completes_interrupted_diagnostic_phases(tmp_path, phase):
    manifest = prepared_fixture(tmp_path / "data")
    config = {
        **configuration(),
        "epochs": 5,
        "probe_interval": 1,
        "early_stopping": {"patience": 1, "min_delta": 1e6},
    }
    whole, partial, resumed = (tmp_path / name for name in ("whole", "partial", "resumed"))
    run(config, manifest, whole)

    class StopInProbe(NoGpuBudget):
        def check(self):
            if (partial / "probes" / phase).exists():
                raise BudgetStop("interrupted diagnostic phase")

    with pytest.raises(BudgetStop):
        run(config, manifest, partial, budget=StopInProbe())
    run(config, manifest, resumed, resume=partial / "last.pth")
    left = torch.load(whole / "last.pth", weights_only=True)
    right = torch.load(resumed / "last.pth", weights_only=True)
    assert left["state"] == right["state"]
    assert right["state"]["epoch"] == 2
    assert right["state"]["pending_probe"] is None
    for name, value in left["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)
    for epoch in ("1", "2", "selected"):
        assert (resumed / "probes" / epoch / "diagnostics.json").exists()
    assert (resumed / "attention.json").exists()
