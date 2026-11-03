"""A study matrix can span immutable queues while sharing one data release."""

import hashlib
import json
import os
from pathlib import Path
import select


def planned_ids(study):
    ids = {f"{domain}--{task}--{stage}--s{seed}"
           for domain, stages in (("in_domain", ("pretrain", "finetune")),
                                  ("cross_domain", ("linear_probe", "finetune")))
           for task in study[domain] for stage in stages for seed in study["training_seeds"]}
    source = study["source"]
    ids.add(f"cross_domain--{source['task']}--pretrain--s{source['seed']}")
    if len(ids) != study["expected_runs"]:
        raise ValueError("Study matrix count does not match its declared run count")
    return ids


def retain_new_tasks(manifest, study, previous_path=None):
    """Retain old queue ownership and generate only missing task identities."""
    expected = planned_ids(study)
    if {t["id"] for t in manifest["tasks"]} != expected:
        raise ValueError("Generated tasks differ from the paired study matrix")
    manifest["study"] = study
    if previous_path is not None:
        path = Path(previous_path).resolve()
        previous = json.loads(path.read_text())
        if previous.get("previous_queue"):
            raise ValueError("Extend the original queue once; do not chain partial studies")
        old = {t["id"] for t in previous["tasks"]}
        if not old <= expected or previous["data_release"] != manifest["data_release"]:
            raise ValueError("Previous queue must be a subset using the exact same data release")
        manifest["previous_queue"] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tasks": previous["tasks"], "config_hashes": previous["config_hashes"],
        }
        removed = [t for t in manifest["tasks"] if t["id"] in old]
        manifest["tasks"] = [t for t in manifest["tasks"] if t["id"] not in old]
        for task in removed:
            for config in [task["config"]] + [p["config"] for p in task.get("placements", [])]:
                Path(config).unlink()
                manifest["config_hashes"].pop(config)
    manifest["expected_tasks"] = len(manifest["tasks"])
    validate_study(manifest)


def validate_study(manifest):
    from pretrain.train import read_config

    study = manifest["study"]
    previous = manifest.get("previous_queue", {})
    if previous:
        path = Path(previous["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous["sha256"]:
            raise ValueError("Previous queue evidence changed")
    old = previous.get("tasks", [])
    tasks = old + manifest["tasks"]
    if len(tasks) != study["expected_runs"] or {t["id"] for t in tasks} != planned_ids(study):
        raise ValueError("Combined queues must cover the paired study exactly once")
    memberships = {}
    for task in tasks:
        path = Path(task["config"])
        if task in old and hashlib.sha256(path.read_bytes()).hexdigest() != previous["config_hashes"][str(path)]:
            raise ValueError("Previous queue configuration changed")
        config = read_config(path)
        if config.get("pretraining_scope") == "tuab_source":
            continue
        name = Path(config["dataset"]).stem
        identity = (str(Path(config["dataset"]).resolve()), config["split_seed"])
        if memberships.setdefault(name, identity) != identity:
            raise ValueError("Domains must use identical dataset files and saved split indices")


def wait_for_previous(manifest, stopped):
    """The immutable old launcher cannot share admission; wait for its completion."""
    previous = manifest.get("previous_queue")
    if not previous:
        return True
    import ctypes

    root = Path(previous["path"]).parent
    receipt = root / "finished.json"
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Cannot watch the original queue completion")
    try:
        # Atomic JSON publication renames a file into this directory.
        if libc.inotify_add_watch(descriptor, os.fsencode(root), 0x80 | 0x8 | 0x100) < 0:
            raise OSError(ctypes.get_errno(), "Cannot watch the original queue directory")
        while not stopped():
            if receipt.is_file():
                state = json.loads(receipt.read_text())
                if (state["exit_code"] != 0 or state.get("stopped")
                        or state["counts"] != {"complete": len(previous["tasks"])}):
                    raise RuntimeError("Original queue did not finish successfully; preserve pending tasks")
                return True
            if select.select([descriptor], [], [], 1)[0]:
                os.read(descriptor, 65536)
        return False
    finally:
        os.close(descriptor)
