"""Bounded performance probes must not compete with paper-queue admission."""

import fcntl
import json
import multiprocessing
from contextlib import ExitStack

import pytest

from pretrain.pipeline.budget import Budget
from scripts.profile_training import benchmark_window, profiling_budget, profile_stages


def ledger(root, *, active=False, meters=None):
    root.mkdir()
    (root / "budget.json").write_text(
        json.dumps(
            {
                "active": active,
                "used_seconds": 100,
                "limit_seconds": 259200,
                "external_meters": meters or {},
            }
        )
    )
    return root


def test_profiler_refuses_a_live_dispatch_queue_before_work(tmp_path):
    root = ledger(tmp_path / "study")
    original = (root / "budget.json").read_text()
    with (root / "queues.lock").open("a") as queue:
        fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="live paper queue"):
            with benchmark_window(root):
                pytest.fail("Profiler entered a live dispatch window")
    assert (root / "budget.json").read_text() == original


def test_profiler_holds_dispatch_lock_until_measurement_finishes(tmp_path):
    root = ledger(tmp_path / "study")
    with benchmark_window(root):
        with (root / "queues.lock").open("a") as queue:
            with pytest.raises(BlockingIOError):
                fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (root / "queues.lock").open("a") as queue:
        fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_profiler_refuses_an_active_device_meter(tmp_path):
    meter = ledger(tmp_path / "meter", active=True)
    root = ledger(tmp_path / "study", meters={str(meter): 12})
    with pytest.raises(RuntimeError, match="GPU work is active"):
        with benchmark_window(root):
            pytest.fail("Profiler ignored an active auxiliary device")


def test_profiler_rejects_an_auxiliary_budget_as_an_independent_allowance(tmp_path):
    root = ledger(tmp_path / "study")
    meter = ledger(root / "meter")
    payload = json.loads((root / "budget.json").read_text())
    payload["external_meters"] = {str(meter): 12}
    (root / "budget.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="authoritative GPU ledger"):
        with benchmark_window(meter):
            pytest.fail("Profiler accepted an independent meter allowance")


def pending_paper(root, channel):
    channel.send("attempting")
    with Budget(root, max_concurrent=4, device="GPU-one").lease():
        channel.send("admitted")


def test_reserved_probe_makes_pending_paper_wait_instead_of_failing(tmp_path, monkeypatch):
    root = tmp_path / "study"
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=pending_paper, args=(root, child))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one")
    try:
        with ExitStack() as papers:
            for _ in range(3):
                papers.enter_context(Budget(root, max_concurrent=4, device="GPU-one").lease())
            with profiling_budget(
                Budget, root, reserve_live=True, max_seconds=10, concurrency=4
            ) as probe:
                probe.flush()
                assert len(json.loads((root / "budget.json").read_text())["leases"]) == 4
                process.start()
                assert parent.poll(10) and parent.recv() == "attempting"
                assert not parent.poll(0.05)
            assert parent.poll(10) and parent.recv() == "admitted"
            process.join(timeout=10)
            assert process.exitcode == 0
        saved = json.loads((root / "budget.json").read_text())
        assert not saved["active"] and not saved["leases"]
        assert saved["limit_seconds"] == 259200
    finally:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)


def test_reserved_probe_rejects_a_window_near_the_shared_cap(tmp_path):
    root = ledger(tmp_path / "study")
    state = json.loads((root / "budget.json").read_text())
    state["used_seconds"] = state["limit_seconds"] - 1
    (root / "budget.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="Insufficient shared budget"):
        with profiling_budget(Budget, root, reserve_live=True, max_seconds=10, concurrency=4):
            pytest.fail("Profiler reserved more than the remaining budget")


def test_stage_annotations_preserve_a_real_update_and_restore_hooks():
    import copy
    import torch
    from pretrain.pipeline import training
    from pretrain.channel_identity import load_channel_vocabulary
    from pretrain.models.factory import create_model, masks
    from pretrain.objectives import DEFAULT_HYDRA

    config = training.resolve_training_config(
        {
            "width": 8,
            "depth": 1,
            "heads": 8,
            "dropout": 0.0,
            "input_samples": 64,
            "objective": {**DEFAULT_HYDRA, "groups": 2, "k": 4},
        }
    )
    original = create_model(config)
    candidate = copy.deepcopy(original)
    values = torch.randn(2, 3, 64)
    ids = torch.tensor([c.index for c in load_channel_vocabulary().channels[:3]])
    mask = masks(["a", "b"], ids, samples=64, seed=241)
    optimizer = training.optimizer_for(original)
    rng = torch.get_rng_state()
    baseline = original(values, ids, mask=mask)["loss"]
    baseline.backward()
    torch.nn.utils.clip_grad_norm_(original.parameters(), 5.0)
    optimizer.step()
    consumed_rng = torch.get_rng_state()
    backward, factory = torch.Tensor.backward, training.optimizer_for

    class ProbeBudget:
        def check(self):
            pass

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        with profile_stages(training, torch, ProbeBudget(), enabled=True) as tracker:
            tracker.parameters(candidate)
            optimizer = training.optimizer_for(candidate)
            torch.set_rng_state(rng)
            measured = candidate(values, ids, mask=mask)["loss"]
            measured.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5.0)
            optimizer.step()
            assert torch.equal(torch.get_rng_state(), consumed_rng)
    assert torch.Tensor.backward is backward and training.optimizer_for is factory
    torch.testing.assert_close(measured, baseline)
    for left, right in zip(original.parameters(), candidate.parameters()):
        torch.testing.assert_close(left, right)
    keys = {event.key for event in trace.key_averages()}
    assert {
        "stage/encoder",
        "stage/decoder",
        "stage/dictionary_target",
        "stage/dictionary_prediction",
        "stage/dictionary_backward",
        "stage/backward_total",
        "stage/optimizer",
    } <= keys
