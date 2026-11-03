"""Prepare a new queue attempt while preserving completed runs and resume identities."""

import argparse
import copy
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_settings(task, state):
    placement = state.get("placements", {}).get(task["id"], task)
    return yaml.safe_load(Path(placement["config"]).read_text())


@contextmanager
def recovery_guard(manifest):
    """Exclude the original launcher and all of its task owners during recovery."""
    with ExitStack() as ownership:
        if manifest.get("recovery"):
            path = Path(manifest["recovery"]["manifest"])
            original = json.loads(path.read_text())
            locks = [path.parent / "launch.lock"]
            for task in original["tasks"]:
                identity = hashlib.sha256(str(Path(task["output"]).resolve()).encode()).hexdigest()
                locks.append(Path(original["claims"]) / f"{identity}.lock")
            for path in locks:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = ownership.enter_context(path.open("a"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from scripts.paper.process import open_matching_process

            for task in original["tasks"]:
                identity = hashlib.sha256(str(Path(task["output"]).resolve()).encode()).hexdigest()
                receipt = Path(original["claims"]) / f"{identity}.json"
                if not receipt.exists():
                    continue
                owner = json.loads(receipt.read_text())
                handle = open_matching_process(owner.get("pid"), owner.get("process_start"))
                if handle is not None:
                    os.close(handle)
                    raise RuntimeError("An original training child is still alive; do not duplicate it")
        yield


def validate_recovery(manifest):
    """Bind changed operational paths to the immutable queue being recovered."""
    from pretrain.train import read_config

    recovery = manifest["recovery"]
    path = Path(recovery["manifest"])
    if digest(path) != recovery["sha256"]:
        raise ValueError("Recovery source manifest changed")
    old = json.loads(path.read_text())
    if old["software_contract"] != manifest["software_contract"]:
        raise ValueError("Recovery must retain the original training software")
    if ({t["id"] for t in old["tasks"]} != {t["id"] for t in manifest["tasks"]}
            or old["data_release"] != manifest["data_release"]):
        raise ValueError("Recovery must retain every original task and the data release")
    originals = {t["id"]: t for t in old["tasks"]}
    operational = {"output", "resume", "budget", "checkpoint"}
    for task in manifest["tasks"]:
        evidence = recovery["tasks"][task["id"]]
        config_path = Path(evidence["config"])
        if (str(config_path) not in old["config_hashes"]
                or digest(config_path) != old["config_hashes"][str(config_path)]):
            raise ValueError("Recovery original configuration changed")
        original = read_config(config_path)
        actual = read_config(task["config"])
        if ({k: v for k, v in original.items() if k not in operational}
                != {k: v for k, v in actual.items() if k not in operational}):
            raise ValueError("Recovery changed the scientific configuration or GPU placement")
        if original["output"] != originals[task["id"]]["output"]:
            raise ValueError("Recovery evidence belongs to another task")
        reference = task.get("completed_output")
        checkpoint = actual.get("resume")
        if reference:
            if reference != original["output"] or checkpoint:
                raise ValueError("Completed recovery reference is inconsistent")
        elif checkpoint:
            if checkpoint != str(Path(original["output"]) / "training/last.pth"):
                raise ValueError("Resume checkpoint must belong to its original task")
            if digest(checkpoint) != evidence["checkpoint_sha256"]:
                raise ValueError("Resume checkpoint changed after preflight")
            receipt = Path(original["output"]) / "wandb.json"
            if digest(receipt) != evidence["wandb_sha256"]:
                raise ValueError("Resume tracking identity changed after preflight")
        elif Path(original["output"]).exists():
            raise ValueError("An interrupted task requires a resume checkpoint")


def checkpoint_preflight(settings):
    """Check checkpoint/config/data/tracking identity without allocating a GPU."""
    import torch
    from pretrain.metadata import fingerprint
    from pretrain.pipeline.checkpoint import dataset_identity, software_contract
    from pretrain.pipeline.dataset import DatasetReference, read_metadata
    from pretrain.pipeline.selection import selection_metadata
    from pretrain.train import resolve_run_config

    payload = torch.load(settings["resume"], map_location="cpu", weights_only=True)
    reference = DatasetReference(
        settings["dataset"], settings.get("split_seed", settings["seed"]),
        settings.get("pretraining_scope", "subject_holdout"))
    metadata = read_metadata(reference)
    config = resolve_run_config(settings, metadata)
    config["gpu_policy"] = {
        "physical_gpu": int(settings["device"].split(":")[1]),
        "shared": settings["shared_gpu"], "concurrent_runs": settings["concurrent_gpu_runs"],
    }
    if settings.get("checkpoint"):
        pretrained = torch.load(settings["checkpoint"], map_location="cpu", weights_only=True)
        recipe = copy.deepcopy(pretrained["config"])
        for key in ("seed", "split_seed", "gpu_policy"):
            recipe.pop(key, None)
        if recipe.get("objective"):
            recipe["objective"].pop("seed", None)
        config["pretraining_recipe_id"] = fingerprint(recipe)[:12]
        receipt = Path(settings["checkpoint"]).parent.parent / "wandb.json"
        if receipt.exists():
            config["source_run_id"] = json.loads(receipt.read_text())["id"]
        config["source_task"] = pretrained["contract"]["dataset_identity"]["task"]
        config["pretraining_checkpoint_sha256"] = digest(settings["checkpoint"])
    config["logical_batch"] = min(config["logical_batch"], metadata["counts"]["train"])
    previous = payload["contract"]
    if config != previous["config"]:
        raise ValueError("Resume configuration differs from the checkpoint")
    if any(previous.get(key) != value for key, value in software_contract().items()):
        raise ValueError("Resume training software or environment changed")
    if (previous["dataset_identity"] != dataset_identity(metadata)
            or previous["dataset"] != metadata["fingerprint"]):
        raise ValueError("Resume data release or saved split changed")
    for key in ("model", "optimizer", "state", "rng"):
        if key not in payload:
            raise ValueError(f"Checkpoint lacks {key}")
    recipe = copy.deepcopy(config)
    for key in ("seed", "split_seed", "purpose", "pretraining_checkpoint_sha256",
                "source_run_id", "gpu_policy"):
        recipe.pop(key, None)
    if recipe.get("objective"):
        recipe["objective"].pop("seed", None)
    recipe_id = fingerprint({"config": recipe, "data": metadata["fingerprint"],
                             "stage": settings["stage"],
                             "selection": selection_metadata(settings["stage"])})[:12]
    receipt = json.loads((Path(settings["resume"]).parent.parent / "wandb.json").read_text())
    expected = {"recipe_id": recipe_id, "seed": settings["seed"], "stage": settings["stage"],
                "domain": settings["wandb"]["domain"],
                "project": settings["wandb"]["project"], "entity": settings["wandb"]["entity"],
                "group": settings["wandb"].get("group")}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("W&B resume identity changed")
    return {"epoch": payload["state"]["epoch"], "cursor": payload["state"]["cursor"],
            "global_step": payload["state"]["global_step"], "wandb_id": receipt["id"],
            "pending_probe": payload["state"].get("pending_probe")}


def prepare(previous_path, root):
    """Freeze a recovery runtime; never overwrite the prior queue or outputs."""
    with recovery_guard({"recovery": {"manifest": str(Path(previous_path).resolve())}}):
        return _prepare(previous_path, root)


def _prepare(previous_path, root):
    previous_path, root = Path(previous_path).resolve(), Path(root).resolve()
    old = json.loads(previous_path.read_text())
    if old.get("recovery"):
        raise ValueError("Recover the original queue; nested recovery queues are not supported")
    subprocess.run([old["python"], str(Path(old["source_root"]) / "scripts/rerun/launch.py"),
                    "--manifest", str(previous_path), "--validate-only"], check=True)
    state = json.loads((previous_path.parent / "queue-state.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    source = root / "source"
    shutil.copytree(old["source_root"], source)
    for relative in ("scripts/rerun/launch.py", "scripts/rerun/recovery.py"):
        shutil.copy2(ROOT / relative, source / relative)
    manifest = copy.deepcopy(old)
    manifest.update(source_root=str(source), claims=str(root / ".claims"),
                    budget=str(root / "usage/gpu0"), config_hashes={}, tasks=[])
    manifest["recovery"] = {"manifest": str(previous_path), "sha256": digest(previous_path),
                            "tasks": {}}
    # Retain the original per-device accounting ledgers and their accumulated usage.
    tasks = {t["id"]: t for t in old["tasks"]}
    completed = {t["id"] for t in old["tasks"] if
                 (Path(t["output"]) / "training/result.json").is_file() and
                 json.loads((Path(t["output"]) / "training/result.json").read_text()).get("status")
                 == "complete"}
    for original_task in old["tasks"]:
        settings = original_settings(original_task, state)
        placement = state.get("placements", {}).get(original_task["id"], original_task)
        task = copy.deepcopy(original_task)
        task.pop("placements", None)
        task["gpu"] = int(settings["device"].split(":")[1])
        original_output = Path(settings["output"])
        evidence = {"config": placement["config"]}
        settings["output"] = str(root / "runs" / original_output.relative_to(previous_path.parent / "runs"))
        settings["resume"] = None
        if task["id"] in completed:
            task["completed_output"] = str(original_output)
        elif original_output.exists():
            if settings["stage"] == "linear_probe":
                raise ValueError("Interrupted independent linear probes require a fresh fit")
            settings["resume"] = str(original_output / "training/last.pth")
            evidence.update(checkpoint_sha256=digest(settings["resume"]),
                            wandb_sha256=digest(original_output / "wandb.json"))
        if task["depends"]:
            parent = tasks[task["depends"][0]]
            parent_output = Path(parent["output"])
            if parent["id"] not in completed:
                parent_output = root / "runs" / parent_output.relative_to(previous_path.parent / "runs")
            settings["checkpoint"] = str(parent_output / "training/best.pth")
        task.update(config=str(root / "configs" / f'{task["id"]}.yaml'), output=settings["output"])
        Path(task["config"]).write_text(yaml.safe_dump(settings, sort_keys=False))
        manifest["config_hashes"][task["config"]] = digest(task["config"])
        manifest["recovery"]["tasks"][task["id"]] = evidence
        manifest["tasks"].append(task)
    manifest["budget"] = old["budget"]
    for relative in ("scripts/rerun/launch.py", "scripts/rerun/recovery.py"):
        manifest["runtime_hashes"][relative] = digest(source / relative)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    prepare(args.previous, args.root)
