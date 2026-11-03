"""Recovery retains checkpoint ownership and rejects changes to the training recipe."""

import copy
import fcntl
import json
from pathlib import Path

import pytest
import yaml

from scripts.rerun.recovery import digest, recovery_guard, validate_recovery


@pytest.fixture
def recovery(tmp_path):
    old = tmp_path / "original"
    old.mkdir()
    config = old / "config.yaml"
    output = old / "runs/task"
    (output / "training").mkdir(parents=True)
    checkpoint = output / "training/last.pth"
    checkpoint.write_bytes(b"checkpoint")
    (output / "wandb.json").write_text('{"id":"existing-run"}')
    settings = {"dataset": str(tmp_path / "data.h5"), "output": str(output),
                "device": "cuda:0", "budget": str(old / "usage/gpu0"), "resume": None,
                "stage": "pretrain", "seed": 241, "microbatch": 256,
                "wandb": {"enabled": True, "project": "eegx", "domain": "in_domain"}}
    config.write_text(yaml.safe_dump(settings))
    task = {"id": "task", "config": str(config), "output": str(output)}
    original = {"tasks": [task], "config_hashes": {str(config): digest(config)},
                "software_contract": {"source": "original"}, "data_release": {},
                "claims": str(old / "claims")}
    manifest = old / "manifest.json"
    manifest.write_text(json.dumps(original))
    new = tmp_path / "queue"
    new.mkdir()
    new_config = new / "config.yaml"
    settings.update(output=str(new / "runs/task"), resume=str(checkpoint))
    new_config.write_text(yaml.safe_dump(settings))
    result = {**copy.deepcopy(original), "tasks": [{**task, "config": str(new_config),
              "output": settings["output"]}], "recovery": {"manifest": str(manifest),
              "sha256": digest(manifest), "tasks": {"task": {"config": str(config),
              "checkpoint_sha256": digest(checkpoint),
              "wandb_sha256": digest(output / "wandb.json")}}}}
    return result


def test_recovery_keeps_original_checkpoint_and_tracking_identity(recovery):
    validate_recovery(recovery)


def test_recovery_rejects_recipe_change(recovery):
    path = Path(recovery["tasks"][0]["config"])
    settings = yaml.safe_load(path.read_text())
    settings["microbatch"] = 512
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="scientific configuration"):
        validate_recovery(recovery)


@pytest.mark.parametrize("name", ["training/last.pth", "wandb.json"])
def test_recovery_rejects_replaced_checkpoint_or_tracking_receipt(recovery, name):
    original = json.loads(Path(recovery["recovery"]["manifest"]).read_text())
    (Path(original["tasks"][0]["output"]) / name).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed after preflight"):
        validate_recovery(recovery)


def test_recovery_excludes_original_launcher_and_task_owner(recovery):
    original_path = Path(recovery["recovery"]["manifest"])
    original = json.loads(original_path.read_text())
    claims = Path(original["claims"])
    claims.mkdir()
    import hashlib

    identity = hashlib.sha256(original["tasks"][0]["output"].encode()).hexdigest()
    for path in (original_path.parent / "launch.lock", claims / f"{identity}.lock"):
        with path.open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError), recovery_guard(recovery):
                pytest.fail("Recovery admitted a second owner")


def test_launcher_holds_original_ownership_before_preflight(recovery, tmp_path, monkeypatch):
    from scripts.rerun import launch

    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(recovery))
    lock = Path(recovery["recovery"]["manifest"]).parent / "launch.lock"

    def preflight(*args, **kwargs):
        with lock.open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return 0

    monkeypatch.setattr(launch, "_run", preflight)
    assert launch.run(path, validate_only=True) == 0


def test_recovery_rejects_orphaned_training_child(recovery):
    import hashlib
    import os
    from scripts.paper.process import process_start

    original = json.loads(Path(recovery["recovery"]["manifest"]).read_text())
    claims = Path(original["claims"])
    claims.mkdir()
    output = original["tasks"][0]["output"]
    identity = hashlib.sha256(output.encode()).hexdigest()
    (claims / f"{identity}.json").write_text(json.dumps(
        {"pid": os.getpid(), "process_start": process_start(os.getpid())}))
    with pytest.raises(RuntimeError, match="still alive"), recovery_guard(recovery):
        pytest.fail("Recovery admitted work alongside an orphaned child")
