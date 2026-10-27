"""Prepare raw task recordings with the shared EEG-X2 QuantPlus signal recipe."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json

from pretrain.pipeline.data import PreparedWriter
from pretrain.pipeline.preprocessing import preprocess, profile
from pretrain.pipeline.dataset.schema import portable
from pretrain.pipeline.dataset.catalog import sampling
from pretrain.pipeline.dataset.provenance import load_provenance


def prepared_recordings(adapter, workers):
    """Bound recording memory and preserve source order across CPU filter workers."""
    if type(workers) is not int or workers < 1:
        raise ValueError("Preparation workers must be a positive integer")

    def process(item):
        values = preprocess(item["volts"], item["sample_rate"], adapter.channels)
        item["source"] = {
            **item["source"],
            "native_sample_rate": item["sample_rate"],
            "native_samples": int(item["volts"].shape[1]),
        }
        return values, {k: v for k, v in item.items() if k not in {"volts", "sample_rate"}}

    source = iter(adapter.recordings())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(process, item) for item in islice(source, workers))
        while pending:
            yield pending.popleft().result()
            item = next(source, None)
            if item is not None:
                pending.append(pool.submit(process, item))


def prepare_adapter(adapter, output, *, source_evidence=None, workers=1, reference_proof=None):
    policy = sampling(
        adapter.task, pilot=adapter.label_evidence.get("complete_subject_scope") is False
    )
    if policy.get("requires_row_provenance") and not policy["pilot"] and reference_proof is None:
        raise ValueError(
            "This established catalog requires --reference-proof with original-row lineage"
        )
    writer = PreparedWriter(
        output,
        task=adapter.task,
        window_samples=adapter.window_samples,
        channels=adapter.channels,
        label_evidence=adapter.label_evidence,
        preprocessing=profile(),
        sampling_policy=policy,
    )
    try:
        for values, item in prepared_recordings(adapter, workers):
            writer.add(values, **item)

        def finalize(metadata):
            if reference_proof is not None:
                metadata["reference_provenance"] = load_provenance(reference_proof)
            metadata["label_evidence"] = portable(adapter.label_evidence)
            if "record_audit" in metadata["label_evidence"]:
                metadata["label_evidence"]["record_audit"].sort(
                    key=lambda row: json.dumps(row, sort_keys=True)
                )
            if source_evidence:
                from pretrain.pipeline.splits import attach_source_evidence

                return attach_source_evidence(metadata, source_evidence)
            return metadata

        return writer.finish(finalize=finalize)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=[
            "dreamer_arousal",
            "stew",
            "tuab",
            "bciciv_2a",
            "kalunga",
            "tuev_binary",
            "crowdsourced",
            "physionet_mi",
        ],
        required=True,
    )
    parser.add_argument("--raw", required=True)
    parser.add_argument("--label-artifact")
    parser.add_argument(
        "--label-config", help="Archived experiment configuration for rating conversion"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-proof", help="Verified original-array row lineage JSON")
    parser.add_argument("--workers", type=int, default=1, help="CPU preprocessing workers")
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Explicit STEW/PhysioNet pilot cohort; default is the complete corpus",
    )
    parser.add_argument("--source-evidence", help="Verified non-rASR source-history JSON")
    args = parser.parse_args()
    try:
        if args.subjects is not None and args.task not in {"stew", "physionet_mi"}:
            parser.error("--subjects is only supported for STEW and PhysioNet")
        if args.task == "stew":
            from .adapters.stew import Stew

            adapter = Stew(args.raw, args.subjects)
        elif args.task == "physionet_mi":
            from .adapters.physionet import Physionet

            adapter = Physionet(args.raw, args.subjects)
        elif args.task == "dreamer_arousal":
            from .adapters.dreamer import Dreamer

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

            adapter = Tuev(args.raw, args.label_artifact)
        else:
            from .adapters.crowdsourced import Crowdsourced

            adapter = Crowdsourced(args.raw)
        result = prepare_adapter(
            adapter,
            args.output,
            source_evidence=args.source_evidence,
            workers=args.workers,
            reference_proof=args.reference_proof,
        )
        print(json.dumps({"status": "ready", "task": args.task, "counts": result["counts"]}))
    except (ValueError, OSError) as error:
        parser.exit(2, f"{args.task}: {error}\n")


if __name__ == "__main__":
    main()
