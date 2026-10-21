"""Immutable main matrix and dependency-aware scheduling for the accepted study."""

import json
import fcntl
from pathlib import Path

from pretrain.objectives import DEFAULT_HYDRA
from pretrain.pipeline.data import Windows
from pretrain.pipeline.budget import Budget
from pretrain.pipeline.checkpoint import software_contract
from pretrain.metadata import atomic_json, fingerprint


IN_DOMAIN = {"dreamer_arousal": 256, "stew": 256, "tuab": 1280, "tuev_binary": 256}
TRANSFER = {"crowdsourced": 256, "bciciv_2a": 512, "kalunga": 256}
SEEDS = (241, 242, 243, 244, 245)
DEFAULT_MICROBATCHES = {256: 64, 512: 64, 1280: 16}


def register_dataset(task, item, root):
    from pretrain.pipeline.splits import require_paper_source, split_manifest

    if "preparing" in item:
        return {"status": "preparing", "manifest": str(Path(item["preparing"]).resolve())}
    if "manifest" not in item:
        return {"status": "blocked", "reason": item.get("blocked", "Source evidence is missing")}
    path = Path(item["manifest"]).resolve()
    try:
        metadata = Windows(path, "train", verify_signals=False).metadata
        if metadata["task"] != task or metadata["window_samples"] != (IN_DOMAIN | TRANSFER)[task]:
            raise ValueError("Manifest task or window length mismatch")
        require_paper_source(metadata)
        manifests = {}
        for seed in SEEDS:
            split = split_manifest(path, root / "splits" / task / str(seed), seed)
            view = Windows(split, "train", verify_signals=False)
            manifests[str(seed)] = {
                "manifest": str(split),
                "fingerprint": view.metadata["fingerprint"],
                "train_windows": len(view),
            }
        return {"status": "ready", "manifest": str(path), "seeds": manifests}
    except (ValueError, FileNotFoundError) as error:
        return {"status": "blocked", "reason": str(error), "manifest": str(path)}


def dataset_for(plan, task, seed):
    registration = plan["datasets"][task]
    if registration["status"] != "ready":
        raise ValueError("Task is blocked")
    return registration["seeds"][str(seed)]


def create_plan(datasets, output, *, physical_gpu=1, microbatch=None, budget_root=None):
    """Freeze the EEG-X matrix, independent seed splits, and shared budget."""
    from pretrain.models.factory import GEOMETRIES

    if type(physical_gpu) is not int or physical_gpu != 1:
        raise ValueError("The accepted EEG-X study requires physical GPU 1")
    if microbatch is not None and (type(microbatch) is not int or microbatch < 1):
        raise ValueError("Microbatch must be a positive integer")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    registered = {
        task: register_dataset(
            task, datasets.get(task, {"blocked": "Source evidence is missing"}), root
        )
        for task in IN_DOMAIN | TRANSFER
    }
    jobs = []
    # Complete the primary architecture before spending budget on scaling.
    for size, (depth, width) in GEOMETRIES.items():
        for source in ("dreamer_arousal", "stew", "tuev_binary", "tuab"):
            for seed in SEEDS:
                selected = registered[source].get("seeds", {}).get(str(seed), {})
                prefix = f"{source}-{size}-soft_dict-{seed}"
                config = {
                    "model": "eegx_hybrid_v2",
                    "size": size,
                    "depth": depth,
                    "width": width,
                    "heads": 8,
                    "seed": seed,
                    "input_samples": IN_DOMAIN[source],
                    "epochs": 100,
                    "warmup_epochs": 10,
                    "early_stopping": {"patience": 10, "min_delta": 1e-4},
                    "precision": "bf16",
                    "dropout": 0.1,
                    "compile": False,
                    "deterministic": True,
                    "logical_batch": min(256, selected.get("train_windows", 256)),
                    "microbatch": microbatch or DEFAULT_MICROBATCHES[IN_DOMAIN[source]],
                    "workers": 2,
                    "purpose": "paper_pretraining",
                    "objective": {**DEFAULT_HYDRA, "seed": seed},
                    "probe_interval": 10,
                    "diagnostic_test_probes": True,
                }
                jobs.append(
                    {
                        "id": prefix,
                        "mode": "pretrain",
                        "task": source,
                        "depends_on": None,
                        "config": config,
                    }
                )
                targets = [source, *TRANSFER] if source == "tuab" else [source]
                for target in targets:
                    mode = "finetune" if target == source else "frozen"
                    downstream = {
                        **config,
                        "input_samples": (IN_DOMAIN | TRANSFER)[target],
                        "epochs": 20,
                        "warmup_epochs": 5,
                        "logical_batch": 64,
                        "microbatch": microbatch
                        or DEFAULT_MICROBATCHES[(IN_DOMAIN | TRANSFER)[target]],
                        "purpose": f"paper_{mode}",
                    }
                    jobs.append(
                        {
                            "id": f"{prefix}-{target}-{mode}",
                            "mode": mode,
                            "task": target,
                            "depends_on": prefix,
                            "config": downstream,
                        }
                    )
    ledger = Path(budget_root).resolve() if budget_root else root
    if budget_root and not (ledger / "budget.json").exists():
        raise ValueError("Paper runs must carry an existing shared budget ledger")
    plan = {
        "schema_version": 3,
        "software": software_contract(),
        "gpu_hours": 72,
        "setup_seconds": 300,
        "physical_gpu": physical_gpu,
        "budget_root": str(ledger),
        "datasets": registered,
        "jobs": jobs,
    }
    plan["fingerprint"] = fingerprint(plan)
    atomic_json(root / "plan.json", plan)
    Budget(ledger, gpu_hours=plan["gpu_hours"], setup_seconds=plan["setup_seconds"]).initialize()
    return plan


def read_plan(root):
    plan = json.loads((Path(root) / "plan.json").read_text())
    if plan["fingerprint"] != fingerprint({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Study plan changed")
    if plan.get("schema_version") != 3 or plan.get("software") != software_contract():
        raise ValueError(
            "Study software changed; do not mix code or environments across paired runs"
        )
    plan["datasets"] = resolve_preparing(plan["datasets"], root)
    return plan


def resolve_preparing(datasets, root):
    """Seal a late preparation once, before any task consumes its windows.

    The immutable plan records preparation intent. Separate immutable receipts
    freeze the actual dataset fingerprint when preparation publishes a manifest.
    """
    resolved = dict(datasets)
    registry = Path(root) / "datasets"
    registry.mkdir(exist_ok=True)
    with (registry / "registry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for task, item in datasets.items():
            if item["status"] != "preparing":
                continue
            receipt = registry / f"{task}.json"
            manifest = Path(item["manifest"])
            if receipt.exists():
                registration = json.loads(receipt.read_text())
                if registration["manifest"] != item["manifest"]:
                    raise ValueError("Sealed dataset path differs from the study intent")
                resolved[task] = registration
            elif manifest.exists():
                registration = register_dataset(task, {"manifest": str(manifest)}, Path(root))
                atomic_json(receipt, registration)
                resolved[task] = registration
            elif (manifest.parent / "blocked.json").exists():
                failure = json.loads((manifest.parent / "blocked.json").read_text())
                resolved[task] = {"status": "blocked", "reason": failure["reason"]}
    return resolved


def verify_dataset(plan, task, seed=241):
    registration = dataset_for(plan, task, seed)
    metadata = Windows(registration["manifest"], "train").metadata
    from pretrain.channel_identity import sha256_file

    parent = plan["datasets"][task]["manifest"]
    if sha256_file(parent) != metadata["parent_manifest_sha256"]:
        raise ValueError("Source manifest changed after study planning")
    if (
        metadata["fingerprint"] != registration["fingerprint"]
        or metadata["task"] != task
        or metadata["window_samples"] != (IN_DOMAIN | TRANSFER)[task]
    ):
        raise ValueError("Prepared dataset changed after study planning")
    # Integrity checks do not construct a test view or extract its features.
    for record in metadata["records"]:
        if sha256_file(Path(registration["manifest"]).parent / record["path"]) != record["sha256"]:
            raise ValueError("Prepared recording changed after study planning")
    return metadata


def latest(root, job):
    attempts = sorted((Path(root) / "jobs" / job).glob("attempt-*/status.json"))
    return json.loads(attempts[-1].read_text()) if attempts else None


def resumable_checkpoint(root, job):
    """Device-only deferrals must not hide a saved optimizer boundary."""
    attempts = sorted((Path(root) / "jobs" / job).glob("attempt-*/status.json"), reverse=True)
    for path in attempts:
        receipt = json.loads(path.read_text())
        if receipt["status"] == "budget_stopped":
            checkpoint = Path(receipt["output"]) / "last.pth"
            if checkpoint.is_file():
                return checkpoint
        elif receipt["status"] != "deferred":
            return None
    return None


def runnable(plan, root):
    for job in plan["jobs"]:
        if any(item["status"] == "preparing" for item in plan["datasets"].values()):
            plan = read_plan(root)
        task = plan["datasets"][job["task"]]
        if task["status"] != "ready":
            continue
        previous = latest(root, job["id"])
        if previous and previous["status"] in {"complete", "failed", "running"}:
            continue
        dependency = latest(root, job["depends_on"]) if job["depends_on"] else None
        if job["depends_on"] and (not dependency or dependency["status"] != "complete"):
            continue
        yield job
