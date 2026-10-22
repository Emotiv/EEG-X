"""Import audited release windows without filtering or normalizing them again."""

import argparse
import json
from pathlib import Path

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.artifacts import inspect_artifact
from pretrain.pipeline.data import PreparedWriter, TASK_WINDOWS
from pretrain.pipeline.preprocessing import processed_profile, profile
from pretrain.pipeline.splits import attach_source_evidence


def prepare(contract_path, output):
    """The audited contract supplies field names, physical order, and identities.

    Each release row remains one complete window. Source partitions are retained
    as evidence only; each YAML run assigns subjects for its selected seed.
    """
    path = Path(contract_path).resolve()
    settings = json.loads(path.read_text())
    samples = TASK_WINDOWS[settings["task"]]
    for name in ("label_evidence", "subject_evidence"):
        if settings[name].get("status") != "verified" or not settings[name].get("authority"):
            raise ValueError(f"Verified attributable {name} is required")
    actual = settings["preprocessing"]
    required = (
        processed_profile() if actual.get("name") == processed_profile()["name"] else profile()
    )
    for key, value in required.items():
        if key != "name" and actual.get(key) != value:
            raise ValueError(f"Processed release preprocessing is incompatible: {key}")
    artifact = (path.parent / settings["artifact"]).resolve()
    digest = sha256_file(artifact)
    if digest != settings["artifact_sha256"]:
        raise ValueError("Processed release artifact changed")
    fields = inspect_artifact(artifact)
    if settings["layout"] not in {"NCT", "NTC"}:
        raise ValueError("Release layout must explicitly name sample, channel and time axes")
    label_map = {int(key): int(value) for key, value in settings["label_map"].items()}
    writer = PreparedWriter(
        output,
        task=settings["task"],
        window_samples=samples,
        channels=settings["channels"],
        label_evidence=settings["label_evidence"],
        preprocessing=actual,
    )
    for partition in settings["partitions"]:
        x = fields[partition["signals"]].read(artifact)
        y = fields[partition["labels"]].read(artifact)
        subjects = fields[partition["subjects"]].read(artifact)
        if (
            x.ndim != 3
            or y.ndim != 1
            or subjects.ndim != 1
            or not len(x) == len(y) == len(subjects)
        ):
            raise ValueError("Release signals, labels and participant IDs are not aligned")
        for row in range(len(x)):
            signal = x[row] if settings["layout"] == "NCT" else x[row].T
            if signal.shape != (len(settings["channels"]), samples):
                raise ValueError("Release window or physical channel order is incompatible")
            if int(y[row]) != y[row] or int(y[row]) not in label_map:
                raise ValueError("Release label is outside the verified mapping")
            subject = subjects[row].item()
            if subject is None or str(subject).strip() in {"", "nan", "None"}:
                raise ValueError("Release participant identity is missing")
            writer.add(
                signal,
                subject=str(subject),
                recording=f"{partition['signals']}:{row}",
                source={
                    "path": str(artifact),
                    "sha256": digest,
                    "release_field": partition["signals"],
                    "release_row": row,
                    "contract_sha256": sha256_file(path),
                },
                label=label_map[int(y[row])],
            )

    def finalize(metadata):
        metadata["subject_evidence"] = settings["subject_evidence"]
        metadata["preprocessing_evidence"] = settings.get(
            "preprocessing_evidence",
            {
                "status": "not_verified",
                "scope": "user-selected processed release; values preserved",
            },
        )
        metadata["release_contract_sha256"] = sha256_file(path)
        metadata["release_counts"] = {
            "input_windows": len(writer.events),
            "retained_windows": sum(metadata["counts"].values()),
            "excluded_entries": len(metadata["excluded"]),
        }
        if settings.get("source_evidence"):
            evidence = (path.parent / settings["source_evidence"]).resolve()
            return attach_source_evidence(metadata, evidence)
        return metadata

    return writer.finish(finalize=finalize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.contract, args.output)
    print(json.dumps({"task": result["task"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
