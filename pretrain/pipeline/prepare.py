"""Prepare eligible raw task data and preserve explicit blocked-task records."""

import argparse
import json
from pathlib import Path

from pretrain.pipeline.data import PreparedWriter
from pretrain.pipeline.preprocessing import preprocess, profile
from pretrain.metadata import atomic_json


def prepare_adapter(adapter, output, *, source_evidence=None):
    writer = PreparedWriter(
        output,
        task=adapter.task,
        window_samples=adapter.window_samples,
        channels=adapter.channels,
        label_evidence=adapter.label_evidence,
        preprocessing=profile(),
    )
    for item in adapter.recordings():
        values = preprocess(item["volts"], item["sample_rate"], adapter.channels)
        item["source"] = {**item["source"], "native_sample_rate": item["sample_rate"]}
        writer.add(values, **{k: v for k, v in item.items() if k not in {"volts", "sample_rate"}})
    metadata = writer.finish()
    if source_evidence:
        from pretrain.pipeline.splits import attach_source_evidence

        metadata = attach_source_evidence(metadata, source_evidence)
        atomic_json(Path(output) / "manifest.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["dreamer_arousal", "tuab", "bciciv_2a", "kalunga", "tuev_binary", "crowdsourced"],
        required=True,
    )
    parser.add_argument("--raw", required=True)
    parser.add_argument("--label-artifact")
    parser.add_argument(
        "--label-config", help="Archived experiment configuration for rating conversion"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-evidence", help="Verified non-rASR source-history JSON")
    args = parser.parse_args()
    try:
        if args.task == "dreamer_arousal":
            from .adapters.dreamer import Dreamer

            if not args.label_artifact:
                parser.error("DREAMER requires an original label artifact for mapping verification")
            adapter = Dreamer(args.raw, args.label_artifact, args.label_config)
        elif args.task == "tuab":
            from .adapters.tuab import Tuab

            adapter = Tuab(args.raw)
        elif args.task == "bciciv_2a":
            from .adapters.bciciv import Bciciv

            adapter = Bciciv(args.raw)
        elif args.task == "kalunga":
            from .adapters.kalunga import Kalunga

            adapter = Kalunga(args.raw)
        elif args.task == "tuev_binary":
            from .adapters.tuev import Tuev

            if not args.label_artifact:
                parser.error(
                    "TUEV requires an archived artifact to verify the native binary event map"
                )
            adapter = Tuev(args.raw, args.label_artifact)
        else:
            from .adapters.crowdsourced import Crowdsourced

            adapter = Crowdsourced(args.raw)
        result = prepare_adapter(adapter, args.output, source_evidence=args.source_evidence)
        print(json.dumps({"status": "ready", "task": args.task, "counts": result["counts"]}))
    except (ValueError, PermissionError, FileNotFoundError) as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        result = {"status": "blocked", "task": args.task, "reason": str(error)}
        atomic_json(output / "blocked.json", result)
        print(json.dumps(result))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
