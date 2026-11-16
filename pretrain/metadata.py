"""Portable identities and atomic metadata for source-native prepared signals."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def subject_key(record: dict) -> str:
    namespace = "emotiv" if record["kind"] == "emotiv_f32" else record["family"]
    return json.dumps([namespace, record["subject"]], separators=(",", ":"))


def allocation_key(record: dict) -> str:
    return (
        "emotiv:" + record["headset"]
        if record["kind"] == "emotiv_f32"
        else "public:" + record["family"]
    )


def segment_identity(record: dict) -> str:
    return fingerprint(
        {
            key: record.get(key)
            for key in (
                "kind",
                "family",
                "headset",
                "subject",
                "recording",
                "path",
                "h5_path",
                "source_range",
                "sha256",
            )
        }
    )
