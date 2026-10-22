"""Exercise source-dataset pairing through the public YAML command."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from pretrain.metadata import fingerprint
from pretrain.pipeline.data import PreparedWriter
from pretrain.pipeline.preprocessing import profile
from pretrain.pipeline.adapters.crowdsourced import CHANNELS
from pretrain.pipeline.adapters.tuab import CHANNELS as TUAB_CHANNELS


ROOT = Path(__file__).resolve().parents[2]


def dataset(root, task, *, release=0):
    samples = 1280 if task == "tuab" else 256
    channels = TUAB_CHANNELS if task == "tuab" else CHANNELS
    writer = PreparedWriter(
        root,
        task=task,
        window_samples=samples,
        channels=channels,
        label_evidence={"status": "verified", "kind": "synthetic_fixture"},
        preprocessing=profile(),
    )
    for subject in range(10):
        values = (
            np.random.default_rng(release * 100 + subject)
            .normal(size=(len(channels), samples * 2))
            .astype("float32")
        )
        writer.add(
            values,
            subject=f"release-{release}-subject-{subject}",
            recording=f"record-{subject}",
            source={"sha256": fingerprint([release, subject, task])},
            events=[(0, 0), (samples, 1)],
        )

    def finalize(metadata):
        metadata["non_rasr_evidence"] = {
            "status": "verified",
            "upstream_rasr": False,
            "authority": "synthetic signal generator",
            "sha256": fingerprint([task, release]),
        }
        return metadata

    writer.finish(finalize=finalize)
    return root / "manifest.json"


def invoke(config, manifest, output, *, stage="pretrain", checkpoint=None):
    config.write_text(
        yaml.safe_dump(
            {
                "manifest": str(manifest),
                "output": str(output),
                "stage": stage,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "device": "cpu",
                "precision": "fp32",
                "epochs": 1,
                "warmup_epochs": 0,
                "depth": 1,
                "width": 16,
                "seed": 241,
                "logical_batch": 14,
                "microbatch": 7,
                "workers": 0,
                "objective": None,
                "probe_interval": 0,
                "optimizer": {"weight_decay": 0.0},
                "wandb": {"enabled": False},
            }
        )
    )
    return subprocess.run(
        [sys.executable, "Main.py", "--config", str(config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )


@pytest.mark.parametrize(
    "stage, task, release, error",
    [
        ("frozen", "crowdsourced", 0, "TUAB"),
        ("finetune", "dreamer_arousal", 1, "dataset and subject split"),
    ],
)
def test_main_rejects_checkpoint_from_the_wrong_paper_source(tmp_path, stage, task, release, error):
    source = dataset(tmp_path / "source", "dreamer_arousal")
    target = dataset(tmp_path / "target", task, release=release)
    config = tmp_path / "run.yaml"
    trained = invoke(config, source, tmp_path / "pretrain")
    assert trained.returncode == 0, trained.stdout + trained.stderr
    output = tmp_path / "evaluation"
    completed = invoke(
        config, target, output, stage=stage, checkpoint=tmp_path / "pretrain/training/best.pth"
    )
    assert completed.returncode != 0, "A mismatched source checkpoint was accepted"
    assert error in completed.stderr
    assert not (output / "training/test_predictions.npz").exists()
    assert not (output / "training/selection.json").exists()


@pytest.mark.parametrize(
    "source_task, target_task, stage",
    [
        ("dreamer_arousal", "dreamer_arousal", "finetune"),
        ("tuab", "crowdsourced", "frozen"),
    ],
)
def test_main_accepts_the_matching_paper_source(tmp_path, source_task, target_task, stage):
    source = dataset(tmp_path / "source", source_task)
    target = source if stage == "finetune" else dataset(tmp_path / "target", target_task)
    config = tmp_path / "run.yaml"
    trained = invoke(config, source, tmp_path / "pretrain")
    assert trained.returncode == 0, trained.stdout + trained.stderr
    output = tmp_path / "evaluation"
    completed = invoke(
        config, target, output, stage=stage, checkpoint=tmp_path / "pretrain/training/best.pth"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "training/result.json").read_text())
    assert result["status"] == "complete"
    identity = result["selection"]["pretraining_contract"]["dataset_identity"]
    assert identity["task"] == source_task
    assert len(identity["release_sha256"]) == len(identity["split_sha256"]) == 64
