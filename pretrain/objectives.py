"""Versioned objective settings shared by EEG-X training."""

import math

DEFAULT_HYDRA = {
    "name": "raw_soft_dict_v2",
    "weight": 1.0,
    "temperature": 1.0,
    "k": 32,
    "groups": 8,
    "kernel_sizes": [9, 15],
    "dilations": [1, 2, 4, 8, 16],
    "seed": 245,
}

DEFAULT_HYDRA_TARGET = {
    "name": "raw_hydra_patch_target_v1",
    "weight": 1.0,
    "patch_size": 64,
    "kernel_size": 15,
    "groups": 4,
    "k": 4,
    "seed": 245,
}


def _validate_target(objective):
    if set(objective) != set(DEFAULT_HYDRA_TARGET):
        raise ValueError("HYDRA targets require explicit patch/filter geometry, weight and seed")
    weight = objective["weight"]
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError("Positive finite objective.weight required")
    for key in ("patch_size", "kernel_size", "groups", "k", "seed"):
        if type(objective[key]) is not int or objective[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid objective.{key}")
    if objective["patch_size"] % 32:
        raise ValueError("HYDRA target patches must contain complete model patches")
    if (objective["kernel_size"] < 3 or objective["kernel_size"] % 2 == 0
            or objective["kernel_size"] > objective["patch_size"]):
        raise ValueError("HYDRA target kernels must be odd and fit inside the target patch")
    if objective["groups"] < 2 or objective["groups"] % 2:
        raise ValueError("HYDRA target groups must be even and at least two")


def validate_objective(objective):
    if objective is None:
        return
    if isinstance(objective, dict) and objective.get("name") == DEFAULT_HYDRA_TARGET["name"]:
        _validate_target(objective)
        return
    if not isinstance(objective, dict) or set(objective) != set(DEFAULT_HYDRA):
        raise ValueError(
            "Soft DiCT requires explicit name, weight, temperature, k, groups, "
            "kernel_sizes, dilations, seed"
        )
    if objective["name"] != DEFAULT_HYDRA["name"]:
        raise ValueError("Unknown objective")
    for key in ("weight", "temperature"):
        value = objective[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Positive finite objective.{key} required")
    for key in ("k", "groups", "seed"):
        if type(objective[key]) is not int or objective[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid objective.{key}")
    if objective["k"] < 2:
        raise ValueError("Soft DiCT requires at least two competing kernels per group")
    for key in ("kernel_sizes", "dilations"):
        values = objective[key]
        if (
            not isinstance(values, (list, tuple))
            or not values
            or any(type(value) is not int or value < 1 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Distinct positive integers required: objective.{key}")
    if any(size < 3 or size % 2 == 0 for size in objective["kernel_sizes"]):
        raise ValueError("Soft DiCT kernel sizes must be odd and at least three")
