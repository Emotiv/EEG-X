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
        root.with_suffix(".h5"),
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
    return writer.path


def invoke(config, manifest, output, *, stage="pretrain", checkpoint=None, **overrides):
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": str(manifest),
                "purpose": "engineering",
                "output": str(output),
                "stage": stage,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "device": "cpu",
                "precision": "fp32",
                "epochs": 1,
                "warmup_epochs": 0,
                "depth": 1,
                "width": 16,
                **({"seed": 241} if "seeds" not in overrides else {}),
                "logical_batch": 14,
                "microbatch": 7,
                "workers": 0,
                "objective": None,
                "probe_interval": 0,
                "optimizer": {"weight_decay": 0.0},
                "wandb": {"enabled": False},
                **overrides,
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
        ("linear_probe", "crowdsourced", 0, "TUAB"),
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
        ("tuab", "crowdsourced", "linear_probe"),
        ("tuab", "crowdsourced", "finetune"),
    ],
)
def test_main_accepts_the_matching_paper_source(tmp_path, source_task, target_task, stage):
    source = dataset(tmp_path / "source", source_task)
    target = source if source_task == target_task else dataset(tmp_path / "target", target_task)
    config = tmp_path / "run.yaml"
    extra = {"pretraining_scope": "tuab_source"} if source_task == "tuab" else {}
    trained = invoke(config, source, tmp_path / "pretrain", **extra)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    output = tmp_path / "evaluation"
    completed = invoke(
        config,
        target,
        output,
        stage=stage,
        checkpoint=tmp_path / "pretrain/training" / ("best.pth"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "training/result.json").read_text())
    assert result["status"] == "complete"
    identity = result["selection"]["pretraining_contract"]["dataset_identity"]
    assert identity["task"] == source_task
    assert len(identity["release_sha256"]) == len(identity["split_sha256"]) == 64
    paper_output = tmp_path / "paper-evaluation"
    rejected = invoke(
        config,
        target,
        paper_output,
        stage=stage,
        checkpoint=tmp_path / "pretrain/training/best.pth",
        purpose="paper",
    )
    assert rejected.returncode != 0
    assert "checkpoint from verified paper pretraining" in rejected.stderr
    assert not (paper_output / "training/test_predictions.npz").exists()
    assert not (paper_output / "wandb.json").exists()


def test_fused_norm_checkpoint_transfers_to_probe_and_finetune(tmp_path):
    import torch

    from pretrain.pipeline.downstream import classifier

    source = dataset(tmp_path / "source", "dreamer_arousal")
    config = tmp_path / "run.yaml"
    trained = invoke(config, source, tmp_path / "pretrain", normalization_backend="triton")
    assert trained.returncode == 0, trained.stdout + trained.stderr
    checkpoint = tmp_path / "pretrain/training/best.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["config"]["normalization_backend"] == "triton"
    assert payload["model"]["_extra_state"]["normalization_backend"] == "triton"
    model, _ = classifier(checkpoint, 2, 241)
    assert model.encoder.normalization_backend == "triton"
    assert model.head[0].backend == "triton"
    for stage in ("linear_probe", "finetune"):
        completed = invoke(config, source, tmp_path / stage, stage=stage, checkpoint=checkpoint)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads((tmp_path / stage / "training/result.json").read_text())
        assert result["status"] == "complete"
        assert result["selection"]["pretraining_contract"]["config"]["normalization_backend"] == "triton"


def test_downstream_backend_option_applies_to_an_existing_native_checkpoint(tmp_path):
    import torch

    from pretrain.models.attention import MultiHeadAttention
    from pretrain.normalization import RMSNorm
    from pretrain.pipeline.dataset import DatasetReference, Windows
    from pretrain.pipeline.downstream import classifier

    source = dataset(tmp_path / "source", "dreamer_arousal")
    config = tmp_path / "run.yaml"
    trained = invoke(config, source, tmp_path / "pretrain")
    assert trained.returncode == 0, trained.stdout + trained.stderr
    checkpoint = tmp_path / "pretrain/training/best.pth"
    original, _ = classifier(checkpoint, 2, 241)
    selected, _ = classifier(checkpoint, 2, 241, normalization_backend="triton")
    assert selected.encoder.normalization_backend == "triton"
    assert all(module.backend == "triton" for module in selected.modules() if isinstance(module, RMSNorm))
    assert all(module.normalization_backend == "triton" for module in selected.modules()
               if isinstance(module, MultiHeadAttention) and module.qk_normalization == "rmsnorm")
    signal = torch.randn(1, len(CHANNELS), 256, generator=torch.Generator().manual_seed(241))
    ids = Windows(DatasetReference(source, 241, "subject_holdout"), "train", verify_signals=False).ids
    original.eval()
    selected.eval()
    with torch.inference_mode():
        torch.testing.assert_close(original(signal, ids), selected(signal, ids), rtol=0, atol=0)
    for stage in ("linear_probe", "finetune"):
        completed = invoke(config, source, tmp_path / stage, stage=stage, checkpoint=checkpoint,
                           normalization_backend="triton")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads((tmp_path / stage / "training/result.json").read_text())
        assert result["status"] == "complete"
        assert result["selection"]["contract"]["config"]["normalization_backend"] == "triton"


def test_tuab_source_encoder_is_shared_by_target_seeds_in_both_downstream_modes(tmp_path):
    source = dataset(tmp_path / "source", "tuab")
    target = dataset(tmp_path / "target", "crowdsourced")
    config = tmp_path / "run.yaml"
    trained = invoke(
        config, source, tmp_path / "pretrain", pretraining_scope="tuab_source", epochs=2
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    manifest = json.loads((tmp_path / "pretrain/split.json").read_text())
    assert manifest["counts"] == {"train": 16, "validation": 4, "test": 0}
    assert set(manifest["subjects"].values()) == {"train", "validation"}
    result = json.loads((tmp_path / "pretrain/training/result.json").read_text())
    assert result["selection"] == "minimum_validation_loss"
    assert result["state"]["epoch"] == 2
    assert all("validation_mse" in row for row in result["state"]["history"])
    checkpoint = tmp_path / "pretrain/training/best.pth"
    assert checkpoint.exists() and not (checkpoint.parent / "final.pth").exists()
    for stage in ("linear_probe", "finetune"):
        completed = invoke(
            config,
            target,
            tmp_path / stage,
            stage=stage,
            checkpoint=checkpoint,
            seeds=[242, 243],
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        for seed in (242, 243):
            root = tmp_path / stage / f"seed-{seed}"
            result = json.loads((root / "training/result.json").read_text())
            assert result["status"] == "complete"
            assert result["selection"]["status"] == "locked_before_test"
            assert result["selection"]["pretraining_contract"]["config"]["seed"] == 241
            assert (
                result["selection"]["pretraining_contract"]["dataset_identity"]["scope"]
                == "tuab_source"
            )
            split = json.loads((root / "split.json").read_text())
            assert split["split_seed"] == seed


def test_checkpoint_contract_detects_an_actual_triton_runtime_change(monkeypatch):
    from importlib import metadata
    from pretrain.pipeline.checkpoint import software_contract

    original = metadata.version
    before = software_contract()

    def changed_version(name):
        return "999.0.0" if name == "triton" else original(name)

    monkeypatch.setattr(metadata, "version", changed_version)
    after = software_contract()
    assert before != after
    assert after["triton"] == "999.0.0"
