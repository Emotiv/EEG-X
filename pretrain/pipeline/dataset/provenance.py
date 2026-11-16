"""Embed and verify established original-array lineage without signal sidecars."""

from collections import Counter
import json
from pathlib import Path

from pretrain.metadata import fingerprint
from .schema import portable


def load_provenance(path):
    proof = portable(json.loads(Path(path).read_text()))
    proof.pop("fingerprint", None)
    proof["mappings"].sort(key=lambda row: (row["recording"], row["start"], row["label"]))
    proof["sources"].sort(key=lambda row: row["file"])
    for row in proof["mappings"]:
        row["original_rows"].sort(key=lambda old: (old["field"], old["row"]))
    proof["fingerprint"] = fingerprint(proof)
    return proof


def validate_provenance(metadata, index):
    proof = metadata.get("reference_provenance")
    if proof is None:
        if metadata.get("sampling", {}).get("requires_row_provenance"):
            raise ValueError("Original-row provenance is required for this reference catalog")
        return
    if (
        proof.get("schema_version") != 1
        or proof.get("task") != metadata["task"]
        or proof.get("status") != "numerically_verified"
        or not proof.get("signal_multiset_matches")
        or proof.get("fingerprint")
        != fingerprint({key: value for key, value in proof.items() if key != "fingerprint"})
    ):
        raise ValueError("Invalid original-row provenance")
    artifact_hash = proof.get("artifact_sha256", "")
    if len(artifact_hash) != 64 or any(c not in "0123456789abcdef" for c in artifact_hash):
        raise ValueError("Original reference artifact hash is missing")
    partitions = proof.get("original_partitions", {})
    if (
        set(partitions) != {"X_train", "X_val", "X_test"}
        or any(type(count) is not int or count < 0 for count in partitions.values())
        or sum(partitions.values()) != len(index)
    ):
        raise ValueError("Original-row provenance lacks complete original array lengths")
    records = {record["recording"]: record for record in metadata["records"]}
    sources = {
        record["source"]["path"]: record["source"]["sha256"] for record in metadata["records"]
    }
    if {item["file"]: item["sha256"] for item in proof["sources"]} != sources:
        raise ValueError("Original-row provenance identifies different raw sources")
    expected, original_rows = Counter(), set()
    for mapping in proof["mappings"]:
        record = records.get(mapping["recording"])
        if record is None or not mapping["original_rows"]:
            raise ValueError("Original-row provenance has an unknown recording or missing row")
        native_start, native_stop = mapping["native_crop"]
        rate = record["source"].get("native_sample_rate", metadata["sample_rate"])
        native_samples = record["source"].get("native_samples")
        expected_length = metadata["sampling"].get("reference_native_samples")
        if (
            type(native_samples) is not int
            or native_samples < 1
            or type(expected_length) is not int
            or expected_length < 1
            or native_stop - native_start != expected_length
            or native_stop > native_samples
            or round(native_samples * 128 / rate) != record["samples"]
        ):
            raise ValueError("Original-row provenance has an invalid native crop length or bound")
        if (
            native_start < 0
            or native_stop <= native_start
            or mapping["start"] != round(native_start * 128 / rate)
        ):
            raise ValueError("Original-row provenance has incompatible crop timing")
        expected[mapping["recording"], mapping["start"], mapping["label"]] += 1
        for old in mapping["original_rows"]:
            if (
                old["field"] not in {"X_train", "X_val", "X_test"}
                or type(old["row"]) is not int
                or old["row"] < 0
                or old["row"] >= partitions[old["field"]]
            ):
                raise ValueError("Original-row provenance has an invalid array row")
            original_rows.add((old["field"], old["row"]))
    actual = Counter(
        (metadata["records"][int(row["record"])]["recording"], int(row["start"]), int(row["label"]))
        for row in index
    )
    if actual != expected or len(original_rows) != len(index):
        raise ValueError("Original-row provenance does not cover the complete sample catalog")
