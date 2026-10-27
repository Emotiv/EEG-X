"""Serializable random states for reproducible optimizer updates."""

import random

import numpy as np
import torch


def capture_rng_state():
    numpy_rng = np.random.get_state(legacy=True)
    if not isinstance(numpy_rng, tuple):
        raise TypeError("Expected legacy NumPy RNG state")
    return {
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": {
            "algorithm": numpy_rng[0],
            "keys": numpy_rng[1].tolist(),
            "position": numpy_rng[2],
            "has_gauss": numpy_rng[3],
            "cached_gaussian": numpy_rng[4],
        },
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }
