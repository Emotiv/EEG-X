import json

import pytest

from pretrain.pipeline.budget import Budget, BudgetStop


def test_production_budget_charges_setup_once_and_reports_it(tmp_path, monkeypatch):
    root = tmp_path / "study"
    Budget(root, setup_seconds=300).initialize()
    saved = json.loads((root / "budget.json").read_text())
    assert saved["used_seconds"] == saved["setup_seconds"] == 300
    clock = [10.0]
    monkeypatch.setattr("pretrain.pipeline.budget.time.perf_counter", lambda: clock[0])
    for _ in range(2):
        with Budget(root, setup_seconds=300).lease():
            clock[0] += 1
    saved = json.loads((root / "budget.json").read_text())
    assert saved["used_seconds"] == 302
    assert saved["setup_seconds"] == 300


def test_existing_ledger_is_preserved_when_setup_is_already_charged(tmp_path):
    root = tmp_path / "study"
    Budget(root, setup_seconds=300).initialize()
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
