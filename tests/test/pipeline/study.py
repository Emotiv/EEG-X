import json

import pytest

from pretrain.pipeline.budget import Budget, BudgetStop
from pretrain.pipeline.reporting import report
from pretrain.pipeline.study import create_plan, read_plan, runnable, verify_dataset
from pretrain.metadata import fingerprint
from tests.test.pipeline.data import prepared_fixture


def test_study_freezes_the_selected_physical_gpu(tmp_path):
    plan = create_plan({}, tmp_path / "gpu-one", physical_gpu=1)
    assert plan["physical_gpu"] == 1
    assert read_plan(tmp_path / "gpu-one")["physical_gpu"] == 1
    with pytest.raises(ValueError, match="physical GPU 1"):
        create_plan({}, tmp_path / "invalid", physical_gpu=-1)


def test_study_records_faster_microbatches_and_an_explicit_override(tmp_path):
    plan = create_plan({}, tmp_path / "defaults")
    for job in plan["jobs"]:
        expected = 16 if job["config"]["input_samples"] == 1280 else 64
        assert job["config"]["microbatch"] == expected
    smaller = create_plan({}, tmp_path / "smaller", microbatch=4)
    assert {job["config"]["microbatch"] for job in smaller["jobs"]} == {4}
    with pytest.raises(ValueError, match="positive integer"):
        create_plan({}, tmp_path / "invalid", microbatch=0)


def test_main_matrix_and_blocked_data_are_explicit(tmp_path):
    plan = create_plan({}, tmp_path / "study")
    assert len(plan["jobs"]) == 330
    assert sum(job["mode"] == "pretrain" for job in plan["jobs"]) == 120
    assert plan["jobs"][0]["config"]["size"] == "d8w64"
    assert {job["config"]["depth"] for job in plan["jobs"]} == {4, 8}
    assert {job["config"]["objective"]["groups"] for job in plan["jobs"]} == {8}
    assert {job["config"]["objective"]["k"] for job in plan["jobs"]} == {32}
    assert not list(runnable(plan, tmp_path / "study"))
    result = report(tmp_path / "study")
    assert result["counts"] == {"blocked": 330}
    assert result["five_seed_groups"] == 0
    path = tmp_path / "study" / "plan.json"
    changed = json.loads(path.read_text())
    changed["gpu_hours"] = 100
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed"):
        read_plan(tmp_path / "study")


def test_production_budget_charges_setup_once_and_reports_it(tmp_path, monkeypatch):
    root = tmp_path / "study"
    create_plan({}, root)
    saved = json.loads((root / "budget.json").read_text())
    assert saved["used_seconds"] == saved["setup_seconds"] == 300
    assert report(root)["budget"]["used_seconds"] == 300
    clock = [10.0]
    monkeypatch.setattr("pretrain.pipeline.budget.time.perf_counter", lambda: clock[0])
    for _ in range(2):
        with Budget(root, setup_seconds=300).lease():
            clock[0] += 1
    saved = json.loads((root / "budget.json").read_text())
    assert saved["used_seconds"] == 302
    assert saved["setup_seconds"] == 300
    assert report(root)["budget"]["used_seconds"] == 302


def test_existing_ledger_is_preserved_when_setup_is_already_charged(tmp_path):
    root = tmp_path / "study"
    create_plan({}, root)
    ledger = {"used_seconds": 8102.6, "limit_seconds": 259200, "active": False}
    (root / "budget.json").write_text(json.dumps(ledger))
    assert Budget(root, setup_seconds=300).initialize() == ledger
    assert json.loads((root / "budget.json").read_text()) == ledger


def test_shared_budget_excludes_cpu_and_enforces_cap(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("pretrain.pipeline.budget.time.perf_counter", lambda: clock[0])
    budget = Budget(tmp_path, gpu_hours=1 / 3600)
    with budget.lease():
        budget.check()
        clock[0] += 0.5
        budget.flush()
        with pytest.raises(BlockingIOError):
            with Budget(tmp_path, gpu_hours=1 / 3600).lease():
                pass
        clock[0] += 0.6
        with pytest.raises(BudgetStop):
            budget.check()
    saved = json.loads((tmp_path / "budget.json").read_text())
    assert saved["used_seconds"] == pytest.approx(1.1)
    assert not saved["active"]
    with pytest.raises(BudgetStop):
        with Budget(tmp_path, gpu_hours=1 / 3600).lease():
            pass
    with pytest.raises(ValueError, match="cap"):
        Budget(tmp_path, gpu_hours=73)


def test_valid_replacement_dataset_is_rejected_by_frozen_plan(tmp_path):
    path = prepared_fixture(tmp_path / "data")
    metadata = json.loads(path.read_text())
    metadata["task"] = "dreamer_arousal"
    metadata["fingerprint"] = fingerprint({k: v for k, v in metadata.items() if k != "fingerprint"})
    path.write_text(json.dumps(metadata))
    plan = create_plan({"dreamer_arousal": {"manifest": str(path)}}, tmp_path / "study")
    verify_dataset(plan, "dreamer_arousal")
    metadata["label_evidence"]["revision"] = "changed_valid_dataset"
    metadata["fingerprint"] = fingerprint({k: v for k, v in metadata.items() if k != "fingerprint"})
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="changed after"):
        verify_dataset(plan, "dreamer_arousal")


def test_late_preparation_seals_once_before_any_training(tmp_path):
    root = tmp_path / "study"
    path = tmp_path / "later" / "manifest.json"
    create_plan({"dreamer_arousal": {"preparing": str(path)}}, root)
    assert read_plan(root)["datasets"]["dreamer_arousal"]["status"] == "preparing"
    prepared_fixture(path.parent)
    metadata = json.loads(path.read_text())
    metadata["task"] = "dreamer_arousal"
    metadata["fingerprint"] = fingerprint({k: v for k, v in metadata.items() if k != "fingerprint"})
    path.write_text(json.dumps(metadata))
    sealed = read_plan(root)
    assert sealed["datasets"]["dreamer_arousal"]["status"] == "ready"
    verify_dataset(sealed, "dreamer_arousal")
    metadata["label_evidence"]["changed"] = True
    metadata["fingerprint"] = fingerprint({k: v for k, v in metadata.items() if k != "fingerprint"})
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="changed after"):
        verify_dataset(read_plan(root), "dreamer_arousal")


def test_study_rejects_software_changes_between_paired_runs(tmp_path, monkeypatch):
    root = tmp_path / "study"
    plan = create_plan({}, root)
    monkeypatch.setattr(
        "pretrain.pipeline.study.software_contract",
        lambda: {**plan["software"], "source": "changed"},
    )
    with pytest.raises(ValueError, match="software changed"):
        read_plan(root)
