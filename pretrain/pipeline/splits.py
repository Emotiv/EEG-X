"""Immutable per-seed subject membership over unchanged prepared signals."""

import json
from pathlib import Path

from pretrain.channel_identity import sha256_file
from pretrain.metadata import atomic_json, fingerprint


def select_split(dataset, output, seed, *, scope="subject_holdout"):
    """Read a frozen H5 split; a run saves provenance without copying EEG or indices."""
    from pretrain.pipeline.dataset import DatasetReference, read_metadata

    selected = DatasetReference(Path(dataset), seed, scope)
    metadata = read_metadata(selected)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "split.json", {"dataset": str(selected.path), **metadata})
    return selected


def validate_prepared_input(metadata):
    """Check the declared input contract without requiring upstream rASR proof."""
    from pretrain.pipeline.preprocessing import processed_profile, profile

    expected = (
        processed_profile()
        if metadata["preprocessing"].get("name") == processed_profile()["name"]
        else profile()
    )
    for key, value in expected.items():
        if key != "name" and metadata["preprocessing"].get(key) != value:
            raise ValueError(f"Prepared preprocessing differs from the accepted contract: {key}")


def attach_source_evidence(metadata, evidence_path):
    evidence = json.loads(Path(evidence_path).read_text())
    expected = sorted({r["source"]["sha256"] for r in metadata["records"]})
    if sorted(evidence.get("source_sha256", [])) != expected:
        raise ValueError("Non-rASR evidence does not identify all source artifacts")
    metadata["non_rasr_evidence"] = {**evidence, "sha256": sha256_file(evidence_path)}
    validate_prepared_input(metadata)
    metadata.pop("fingerprint", None)
    metadata["fingerprint"] = fingerprint(metadata)
    return metadata


def validate_run_dataset(config, metadata):
    """Engineering checks cannot silently become paper experiments."""
    if config.get("pretraining_scope") == "tuab_source":
        if (
            metadata["task"] != "tuab"
            or metadata.get("split_scope") != "tuab_source"
            or set(metadata["subjects"].values()) != {"train", "validation"}
            or not metadata["counts"]["validation"]
            or metadata["counts"]["test"]
        ):
            raise ValueError(
                "Shared TUAB pretraining requires separate training and validation subjects"
            )
    elif metadata.get("split_scope") == "tuab_source":
        raise ValueError("Shared TUAB data cannot be used as an in-domain subject holdout")
    if config.get("purpose", "").startswith("paper_"):
        validate_prepared_input(metadata)
        from pretrain.pipeline.dataset.catalog import REFERENCES

        policy = metadata.get("sampling", {})
        status = metadata.get("reference_comparison", {}).get("status")
        if (
            policy.get("name") != "original_catalog_eegx2_v1"
            or policy.get("pilot", True)
            or status not in {"matched", "documented_exception"}
        ):
            raise ValueError("Paper experiments require a verified complete reference catalog")
        if REFERENCES.get(metadata["task"], {}).get("requires_row_provenance") and not metadata.get(
            "reference_provenance"
        ):
            raise ValueError("Paper experiments require established original-row provenance")
        if metadata.get("split_seed") != config["seed"]:
            raise ValueError("Paper experiments require the matching per-seed split")
