"""Proven sample references and explicit exceptions for native regeneration."""

from copy import deepcopy

import numpy as np


REFERENCES = {
    "dreamer_arousal": {"stride_samples": 64, "class_counts": [88611, 81635]},
    "stew": {"stride_samples": 32, "class_counts": [28464, 28464]},
    "tuab": {"stride_samples": 1280, "class_counts": [203616, 205839]},
    "tuev_binary": {"stride_samples": None, "class_counts": [73372, 25312]},
    "bciciv_2a": {
        "stride_samples": None,
        "class_counts": [1296, 1296, 1296, 1296],
        "reference_native_samples": 500,
        "requires_row_provenance": True,
    },
    "kalunga": {
        "stride_samples": None,
        "class_counts": [240, 240, 240, 240],
        "reference_native_samples": 512,
        "requires_row_provenance": True,
    },
    "physionet_mi": {
        "stride_samples": None,
        "requires_row_provenance": True,
        "class_counts": [2455, 2480, 2438, 2465, 9184],
        "reference_native_samples": 320,
        "exception": "harmonize_legacy_320_native_sample_crops_to_true_two_seconds",
    },
    "crowdsourced": {
        "stride_samples": 32,
        "class_counts": [6679, 6684],
        "allow_count_difference": True,
        "exception": "correct_native_timing_and_preserve_acquisition_and_label_boundaries",
    },
}


def sampling(task, *, pilot=False):
    return {
        "name": "original_catalog_eegx2_v1",
        "overlap": "preserved",
        "duplicates": "retained_with_grouped_subject_splits",
        "identity": "record_crop_label_occurrence_v1",
        "pilot": pilot,
        **deepcopy(REFERENCES.get(task, {})),
    }


def validate_reference(metadata, index):
    policy = metadata["sampling"]
    expected = policy.get("class_counts")
    if expected is None:
        return
    actual = np.bincount(index["label"]).tolist()
    metadata["reference_comparison"] = {
        "expected_class_counts": expected,
        "actual_class_counts": actual,
        "status": "pilot"
        if policy["pilot"]
        else "documented_exception"
        if policy.get("exception")
        else "matched",
    }
    if (
        not policy["pilot"]
        and not policy.get("allow_count_difference", False)
        and actual != expected
    ):
        raise ValueError(f"Reference class counts differ: expected {expected}, got {actual}")
