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
