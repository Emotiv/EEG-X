import json
import multiprocessing
import os

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


def test_four_leases_charge_one_gpu_once_and_keep_other_owners(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("pretrain.pipeline.budget.time.perf_counter", lambda: clock[0])
    budgets = [Budget(tmp_path, max_concurrent=4, device="GPU-one") for _ in range(4)]
    with budgets[0].lease():
        clock[0] += 1
        with budgets[1].lease(), budgets[2].lease(), budgets[3].lease():
            with pytest.raises(BlockingIOError):
                with Budget(tmp_path, max_concurrent=4, device="GPU-one").lease():
                    pass
            with pytest.raises(ValueError, match="one GPU"):
                with Budget(tmp_path, max_concurrent=4, device="GPU-two").lease():
                    pass
            clock[0] += 2
            for budget in budgets:
                budget.flush()
            saved = json.loads((tmp_path / "budget.json").read_text())
            assert saved["used_seconds"] == 3
            assert len(saved["leases"]) == 4
        saved = json.loads((tmp_path / "budget.json").read_text())
        assert saved["active"] and len(saved["leases"]) == 1
        clock[0] += 1
    saved = json.loads((tmp_path / "budget.json").read_text())
    assert saved["used_seconds"] == 4
    assert not saved["active"] and not saved["leases"]


def _lease_in_child(root, ready, release, crash):
    with Budget(root, max_concurrent=4, device="GPU-one").lease():
        ready.send(os.getpid())
        release.recv()
        if crash:
            os._exit(1)


def test_four_real_processes_share_ledger_and_recover_a_crashed_owner(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = []
    channels = []
    try:
        for index in range(4):
            parent, child = context.Pipe()
            process = context.Process(
                target=_lease_in_child, args=(tmp_path, child, child, index == 0)
            )
            process.start()
            processes.append(process)
            channels.append(parent)
            assert parent.poll(20), "Child failed to acquire its GPU lease"
            assert parent.recv() == process.pid
        saved = json.loads((tmp_path / "budget.json").read_text())
        assert len(saved["leases"]) == 4
        channels[0].send(True)
        processes[0].join(timeout=20)
        assert processes[0].exitcode == 1
        with Budget(tmp_path, max_concurrent=4, device="GPU-one").lease():
            saved = json.loads((tmp_path / "budget.json").read_text())
            assert len(saved["leases"]) == 4
            assert all(lease["pid"] != processes[0].pid for lease in saved["leases"].values())
        for channel in channels[1:]:
            channel.send(True)
        for process in processes[1:]:
            process.join(timeout=20)
            assert process.exitcode == 0
        saved = json.loads((tmp_path / "budget.json").read_text())
        assert not saved["active"] and not saved["leases"]
        assert 0 < saved["used_seconds"] < 80
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=20)


def test_explicit_uncapped_ledger_keeps_accounting_and_safe_stop(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("pretrain.pipeline.budget.time.perf_counter", lambda: clock[0])
    Budget(tmp_path, gpu_hours=None).initialize()
    budget = Budget.from_ledger(tmp_path)
    with budget.stop_on_signals(), budget.lease():
        clock[0] += 100 * 3600
        budget.check()
        budget.flush()
        budget.stop_requested = True
        with pytest.raises(BudgetStop):
            budget.check()
    state = json.loads((tmp_path / "budget.json").read_text())
    assert state["limit_seconds"] is None
    assert state["used_seconds"] == 100 * 3600
    assert not state["active"]
    with pytest.raises(ValueError, match="extension"):
        Budget(tmp_path).initialize()


def test_from_ledger_preserves_exhausted_old_cap(tmp_path):
    Budget(tmp_path).initialize()
    path = tmp_path / "budget.json"
    state = json.loads(path.read_text())
    state["used_seconds"] = state["limit_seconds"] + 1
    path.write_text(json.dumps(state))
    with pytest.raises(BudgetStop):
        with Budget.from_ledger(tmp_path).lease():
            pass
    assert json.loads(path.read_text())["limit_seconds"] == 72 * 3600
