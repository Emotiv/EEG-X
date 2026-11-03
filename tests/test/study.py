"""Extend the live study without duplicate tasks or changed paired memberships."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from pretrain.train import read_config
from scripts.rerun.prepare import build_tasks
from scripts.rerun.study import planned_ids, retain_new_tasks, validate_study


ROOT = Path(__file__).resolve().parents[2]


def extension(tmp_path):
    configs = {p.stem: read_config(p) for p in (ROOT / "configs").glob("*.yaml")}
    data = tmp_path / "data"
    data.mkdir()
    for config in configs.values():
        (data / Path(config["dataset"]).name).touch()
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    for root in (old_root, new_root):
        (root / "configs").mkdir(parents=True)
    previous = build_tasks({k: v for k, v in configs.items() if "-in-domain" not in k},
                           old_root, 49, data, flexible=True)
    current = build_tasks(configs, new_root, 73, data, flexible=True)
    previous["data_release"] = current["data_release"] = {"release": "unchanged"}
    path = old_root / "manifest.json"
    path.write_text(json.dumps(previous))
    study = json.loads((ROOT / "reproduction/eegx-fixed-v1/study.json").read_text())
    return previous, current, path, study


def test_supplement_adds_only_missing_paired_in_domain_runs(tmp_path):
    previous, current, path, study = extension(tmp_path)
    before = path.read_bytes()
    retain_new_tasks(current, study, path)
    assert current["expected_tasks"] == len(current["tasks"]) == 24
    assert len(planned_ids(study)) == 73
    assert {t["domain"] for t in current["tasks"]} == {"in_domain"}
    assert not ({t["id"] for t in current["tasks"]} & {t["id"] for t in previous["tasks"]})
    assert all(Path(t["config"]).is_file() for t in previous["tasks"])
    assert path.read_bytes() == before
    assert all(set(t["depends"]) <= {p["id"] for p in current["tasks"]} for t in current["tasks"])
    assert len(current["config_hashes"]) == 48


@pytest.mark.parametrize("field", ["split_seed", "dataset"])
def test_paired_domains_cannot_change_the_saved_bank_or_file(tmp_path, field):
    _, current, path, study = extension(tmp_path)
    retain_new_tasks(current, study, path)
    config_path = Path(current["tasks"][0]["config"])
    config = yaml.safe_load(config_path.read_text())
    config[field] = 243 if field == "split_seed" else "/another-release/" + Path(config[field]).name
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="identical dataset"):
        validate_study(current)


def test_existing_evidence_and_duplicate_task_ownership_are_rejected(tmp_path):
    _, current, path, study = extension(tmp_path)
    retain_new_tasks(current, study, path)
    duplicate = copy.deepcopy(current)
    duplicate["tasks"].append(duplicate["previous_queue"]["tasks"][0])
    with pytest.raises(ValueError, match="exactly once"):
        validate_study(duplicate)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence changed"):
        validate_study(current)


def test_supplement_waits_for_successful_original_completion(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from pretrain.metadata import atomic_json
    from scripts.rerun.study import wait_for_previous

    previous, current, path, study = extension(tmp_path)
    retain_new_tasks(current, study, path)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(wait_for_previous, current, lambda: False)
        assert not future.done()
        atomic_json(path.parent / "finished.json", {
            "exit_code": 0, "stopped": False, "counts": {"complete": len(previous["tasks"])}})
        assert future.result(timeout=5)
    atomic_json(path.parent / "finished.json", {
        "exit_code": 1, "stopped": False, "counts": {"failed": 1}})
    with pytest.raises(RuntimeError, match="did not finish"):
        wait_for_previous(current, lambda: False)
    assert wait_for_previous(current, lambda: True) is False
