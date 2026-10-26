"""Exercise the YAML command through real CPU training and checkpoint resume."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from pretrain.train import read_config
from tests.test.pipeline.data import prepared_fixture


ROOT = Path(__file__).resolve().parents[2]


def invoke(config, cwd):
    return subprocess.run(
        [sys.executable, "-m", "pretrain", "train", "--config", str(config)],
        cwd=cwd,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
    )


def test_yaml_command_trains_and_resumes_one_model_without_a_study(tmp_path):
    prepared_fixture(tmp_path / "data")
    path = tmp_path / "train.yaml"
    settings = {
        "dataset": "data.h5",
        "purpose": "engineering",
        "output": "first",
        "device": "cpu",
        "seed": 242,
        "epochs": 1,
        "warmup_epochs": 0,
        "precision": "fp32",
        "logical_batch": 256,
        "microbatch": 7,
        "workers": 0,
        "probe_interval": 0,
        "objective": None,
        "optimizer": {"lr": 0.0004, "eps": 1e-8, "weight_decay": 0.0},
        "scheduler": {"minimum_lr": 0.0002},
        "gradient_clip_norm": 2.0,
    }
    path.write_text(yaml.safe_dump(settings))
    completed = invoke(path, ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = tmp_path / "first"
    assert (output / "training" / "best.pth").is_file()
    assert (output / "training" / "last.pth").is_file()
    assert not (output / "plan.json").exists()
    split = json.loads((output / "split.json").read_text())
    assert split["split_seed"] == 242
    receipt = json.loads((output / "training" / "run.json").read_text())
    assert receipt["resolved"]["optimizer"]["defaults"]["lr"] == 0.0004
    assert receipt["resolved"]["scheduler"]["minimum_lr"] == 0.0002
    assert receipt["resolved"]["gradient_clip_norm"] == 2.0
    payload = torch.load(output / "training" / "last.pth", weights_only=True)
    assert payload["contract"]["config"]["logical_batch"] == 28
    assert payload["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.0002)
    assert payload["contract"]["config"]["size"] == "d8w64"

    # Launch from another directory to ensure YAML-relative paths stay stable.
    resumed = yaml.safe_load((output / "config.yaml").read_text())
    resumed.update(output=str(tmp_path / "resumed"), resume=str(output / "training" / "last.pth"))
    path.write_text(yaml.safe_dump(resumed))
    command = [sys.executable, "-m", "pretrain", "train", "--config", str(path)]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    right = torch.load(tmp_path / "resumed" / "training" / "last.pth", weights_only=True)
    assert payload["state"] == right["state"]
    for name, value in payload["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)
    assert (
        json.loads((tmp_path / "resumed" / "training" / "result.json").read_text())["status"]
        == "complete"
    )


def test_main_script_logs_real_offline_wandb_metrics_and_evaluates(tmp_path):
    prepared_fixture(tmp_path / "data")
    config = tmp_path / "train.yaml"
    settings = {
        "dataset": "data.h5",
        "purpose": "engineering",
        "output": "pretrain",
        "device": "cpu",
        "precision": "fp32",
        "epochs": 1,
        "warmup_epochs": 0,
        "workers": 0,
        "depth": 2,
        "width": 32,
        "objective": None,
        "probe_interval": 0,
        "wandb": {"enabled": True, "project": "eegx-test", "mode": "offline"},
    }
    config.write_text(yaml.safe_dump(settings))
    command = [sys.executable, "Main.py", "--config", str(config)]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "pretrain" / "wandb.json").is_file()
    assert list((tmp_path / "pretrain" / "wandb").glob("offline-run-*/run-*.wandb"))
    settings.update(
        stage="linear_probe", checkpoint="pretrain/training/best.pth", output="evaluation"
    )
    config.write_text(yaml.safe_dump(settings))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((tmp_path / "evaluation" / "training" / "result.json").read_text())
    assert result["selection"]["status"] == "locked_before_test"
    assert {
        "acc",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "f1_score_micro",
        "f1_score_macro",
    } <= result["test"].keys()
    settings.update(seed=242, output="wrong-seed")
    config.write_text(yaml.safe_dump(settings))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )
    assert completed.returncode != 0
    assert "must match the configured seed and model" in completed.stderr
    assert not (tmp_path / "wrong-seed").exists()


def test_yaml_command_accepts_data_without_upstream_rasr_evidence(tmp_path):
    from pretrain.pipeline.dataset import read_metadata

    dataset = prepared_fixture(tmp_path / "data")
    assert "non_rasr_evidence" not in read_metadata(dataset)
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": "data.h5",
                "purpose": "engineering",
                "output": "run",
                "device": "cpu",
                "precision": "fp32",
                "epochs": 1,
                "warmup_epochs": 0,
                "depth": 1,
                "width": 16,
                "workers": 0,
                "objective": None,
                "probe_interval": 0,
                "optimizer": {"weight_decay": 0.0},
            }
        )
    )
    completed = invoke(config, ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "run/training/best.pth").exists()
    saved = json.loads((tmp_path / "run/split.json").read_text())
    assert "non_rasr_evidence" not in saved
    assert not list(tmp_path.rglob("status.json"))


def test_one_main_command_runs_seed_list_locally_with_separate_wandb_runs(tmp_path):
    prepared_fixture(tmp_path / "data")
    config = tmp_path / "seeds.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": "data.h5",
                "purpose": "engineering",
                "output": "seeds",
                "seeds": [241, 242],
                "device": "cpu",
                "precision": "fp32",
                "epochs": 1,
                "warmup_epochs": 0,
                "workers": 0,
                "depth": 2,
                "width": 32,
                "objective": None,
                "probe_interval": 0,
                "wandb": {"enabled": True, "mode": "offline", "project": "eegx-test"},
            }
        )
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "Main.py"), "--config", str(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipts = []
    assignments = []
    for seed in (241, 242):
        output = tmp_path / "seeds" / f"seed-{seed}"
        payload = torch.load(output / "training" / "last.pth", weights_only=True)
        assert payload["contract"]["config"]["seed"] == seed
        receipts.append(json.loads((output / "wandb.json").read_text()))
        assignments.append(json.loads((output / "split.json").read_text())["subjects"])
    assert receipts[0]["id"] != receipts[1]["id"]
    assert receipts[0]["group"] == receipts[1]["group"]
    assert assignments[0] != assignments[1]
    assert completed.stdout.index("Running seed 241 locally") < completed.stdout.index(
        "Running seed 242 locally"
    )


@pytest.mark.parametrize(
    "content, message",
    [
        ("[]", "mapping"),
        ("dataset: data.h5\noutput: run\ndevice: cpu\nshared_gpu: true", "requires a CUDA"),
        ("dataset: data.h5\noutput: run\ndevice: cpu\nshared_gpu: yesplease", "must be boolean"),
        ("dataset: data.h5\noutput: run\ndevice: cpu\nconcurrent_gpu_runs: 0", "positive integer"),
        (
            "dataset: data.h5\noutput: run\ndevice: cpu\nconcurrent_gpu_runs: 4",
            "requires shared_gpu",
        ),
        ("dataset: data.h5\noutput: run\ndevice: cpu\npurpose: research", "purpose must"),
        ("dataset: data.h5\noutput: run\nepoches: 3", "Unknown training settings"),
        ("dataset: data.h5\noutput: run\ndevice: cuda:1", "shared budget"),
        ("dataset: data.h5\noutput: run\nseed: 241\nseeds: [241, 242]", "not both"),
        ("dataset: data.h5\noutput: run\nseeds: [241, 241]", "duplicates"),
        ("dataset: data.h5\noutput: run\nseeds: []", "nonempty list"),
    ],
)
def test_invalid_yaml_fails_before_training(tmp_path, content, message):
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_config(path)
