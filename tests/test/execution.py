import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pretrain.compute import DEFAULT_COMPUTE, configure_compute


@pytest.mark.parametrize(
    "module",
    [
        "pretrain",
        "pretrain.pipeline",
        "pretrain.pipeline.prepare",
        "pretrain.pipeline.processed",
        "pretrain.pipeline.preflight",
        "pretrain.pipeline.benchmark",
    ],
)
def test_public_module_entrypoints_use_local_package(module):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "usage:" in result.stdout


def test_main_entrypoint_exposes_one_yaml_configuration():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "Main.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout
    assert "--study" not in result.stdout


def test_compute_cpu_fallback_and_validation():
    with configure_compute({"compute": DEFAULT_COMPUTE}, "cpu") as effective:
        assert effective["matmul"] == "ieee"
    invalid = copy.deepcopy(DEFAULT_COMPUTE)
    invalid["matmul"] = "unknown"
    with pytest.raises(ValueError):
        with configure_compute({"compute": invalid}, "cpu"):
            pass


def test_allocation_fill_option_is_scoped_and_checkpointed(monkeypatch):
    from pretrain.pipeline import checkpoint

    original_fill = torch.utils.deterministic.fill_uninitialized_memory
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    config = {
        "deterministic": True,
        "compute": {**DEFAULT_COMPUTE, "fill_uninitialized_memory": False},
    }
    with pytest.raises(RuntimeError, match="abort"):
        with configure_compute(config, "cpu") as effective:
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.utils.deterministic.fill_uninitialized_memory is False
            assert effective["fill_uninitialized_memory"] is False
            raise RuntimeError("abort")
    assert torch.utils.deterministic.fill_uninitialized_memory == original_fill
    assert torch.are_deterministic_algorithms_enabled() == original_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == original_warn_only
    with configure_compute(config, "cpu"):
        assert torch.utils.deterministic.fill_uninitialized_memory is False
        with configure_compute(
            {"compute": {**DEFAULT_COMPUTE, "fill_uninitialized_memory": True}}, "cpu"
        ):
            assert torch.utils.deterministic.fill_uninitialized_memory is True
        assert torch.utils.deterministic.fill_uninitialized_memory is False
    assert torch.utils.deterministic.fill_uninitialized_memory == original_fill
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
        with configure_compute({"compute": DEFAULT_COMPUTE}, "cpu"):
            assert torch.utils.deterministic.fill_uninitialized_memory is False
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = original_fill

    monkeypatch.setattr(checkpoint, "software_contract", lambda: {})
    dataset = {
        "task": "tuab", "fingerprint": "release", "subjects": [], "windows_sha256": "windows"
    }
    legacy = checkpoint.contract({"compute": DEFAULT_COMPUTE}, dataset)
    candidate = checkpoint.contract(config, dataset)
    assert "fill_uninitialized_memory" not in legacy["execution"]["requested_compute"]
    assert set(legacy["execution"]["effective_compute"]) == set(DEFAULT_COMPUTE)
    assert candidate["execution"]["requested_compute"]["fill_uninitialized_memory"] is False
    assert candidate["execution"]["effective_compute"]["fill_uninitialized_memory"] is False
    assert candidate != legacy
    for invalid in [0, None, "false"]:
        with pytest.raises(ValueError, match="fill_uninitialized_memory"):
            invalid_config = {"compute": {**DEFAULT_COMPUTE, "fill_uninitialized_memory": invalid}}
            with configure_compute(invalid_config, "cpu"):
                pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_compute_settings_restore_after_error():
    before = (
        torch.backends.fp32_precision,
        torch.backends.cuda.matmul.fp32_precision,
        torch.backends.cudnn.conv.fp32_precision,
        torch.backends.cudnn.benchmark,
    )
    with pytest.raises(RuntimeError, match="abort"):
        with configure_compute({"compute": DEFAULT_COMPUTE, "precision": "bf16"}, "cuda"):
            assert torch.backends.cuda.matmul.fp32_precision == "tf32"
            assert torch.backends.cudnn.conv.fp32_precision == "ieee"
            raise RuntimeError("abort")
    after = (
        torch.backends.fp32_precision,
        torch.backends.cuda.matmul.fp32_precision,
        torch.backends.cudnn.conv.fp32_precision,
        torch.backends.cudnn.benchmark,
    )
    assert before == after
