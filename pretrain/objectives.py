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


def validate_objective(objective):
    if objective is None:
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
            not isinstance(values, (list, tuple)) or not values
            or any(type(value) is not int or value < 1 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Distinct positive integers required: objective.{key}")
    if any(size < 3 or size % 2 == 0 for size in objective["kernel_sizes"]):
        raise ValueError("Soft DiCT kernel sizes must be odd and at least three")
