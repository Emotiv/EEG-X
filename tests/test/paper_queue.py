"""Live expansion keeps accounting additive and executes each output once."""

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import fcntl

import yaml

from pretrain.pipeline.budget import Budget


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/paper"
sys.path.insert(0, str(SCRIPTS))
from accounting import SharedAccounting  # noqa: E402
import task as task_runner  # noqa: E402
from prepare import prepare  # noqa: E402


def test_two_gpu_accounting_survives_legacy_writes_without_double_charge(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("pretrain.pipeline.budget.time.perf_counter", lambda: clock[0])
    primary = Budget(tmp_path / "primary", setup_seconds=300, max_concurrent=4, device="GPU-one")
    meter = Budget(tmp_path / "meter", max_concurrent=4, device="GPU-zero")
    primary.initialize()
    meter.initialize()
    accounting = SharedAccounting(primary.root, meter.root)
    with primary.lease():
        clock[0] += 2
        with meter.lease():
            clock[0] += 3
            state, _ = accounting.settle()
            assert state["used_seconds"] == 308  # Setup plus two and two times three.
            assert len(state["leases"]) == 1
            primary.flush()  # An unchanged writer must retain the external debit.
            accounting.settle()
            assert json.loads(primary.path.read_text())["used_seconds"] == 308
            clock[0] += 1
        clock[0] += 2
        state, _ = accounting.settle()
        assert state["used_seconds"] == 312
    state, _ = accounting.settle()
    assert state["used_seconds"] == 312
    assert state["limit_seconds"] == 259200
    assert not state["active"]
    assert state["external_meters"][str(meter.root.resolve())] == 4


def task_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Main.py").write_text(
        "import pathlib,sys,json,yaml,time,signal\n"
        "s=yaml.safe_load(pathlib.Path(sys.argv[-1]).read_text())\n"
        "out=pathlib.Path(s['output']);out.mkdir()\n"
        "(out/'config.yaml').write_text(yaml.safe_dump(s))\n"
        "(out/'training').mkdir()\n"
        "def stop(*_):\n"
        " (out/'training/result.json').write_text(json.dumps({'status':'budget_stopped'}))\n"
        " sys.exit(2)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "with (out.parent/'executions').open('a') as f: f.write('run\\n')\n"
        "print('READY',flush=True)\n"
        "if s.get('block_file'):\n"
        " with open(s['block_file']) as f: f.read(1)\n"
        "else: time.sleep(0.5)\n"
        "(out/'training/result.json').write_text(json.dumps({'status':'complete'}))\n"
    )
    budget = tmp_path / "budget"
    Budget(budget).initialize()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"source_root": str(source), "budget": str(budget), "claims": str(tmp_path / "claims")}
        )
    )
    config = tmp_path / "task.yaml"
    config.write_text(
        yaml.safe_dump(
            {"output": str(tmp_path / "output"), "device": "cpu", "seed": 241, "stage": "pretrain"}
        )
    )
    command = [
        sys.executable,
        str(SCRIPTS / "task.py"),
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--guardian",
        str(os.getpid()),
    ]
    return command


def test_duplicate_dispatch_executes_one_training_process(tmp_path):
    command = task_fixture(tmp_path)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(subprocess.run, command, capture_output=True, text=True) for _ in range(2)
        ]
        results = [f.result(timeout=30) for f in futures]
    assert sum(r.returncode == 0 for r in results) == 1, [r.stderr for r in results]
    assert "BlockingIOError" in next(r.stderr for r in results if r.returncode)
    assert (tmp_path / "executions").read_text() == "run\n"
    restarted = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert restarted.returncode == 0, restarted.stderr
    assert "SKIP completed" in restarted.stdout
    assert (tmp_path / "executions").read_text() == "run\n"


def test_incomplete_output_is_preserved_and_not_restarted(tmp_path):
    command = task_fixture(tmp_path)
    (tmp_path / "output").mkdir()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Task is incomplete" in result.stderr
    assert not (tmp_path / "executions").exists()


def blocked_fixture(tmp_path):
    command = task_fixture(tmp_path)
    pipe = tmp_path / "release"
    os.mkfifo(pipe)
    settings = yaml.safe_load((tmp_path / "task.yaml").read_text())
    settings["block_file"] = str(pipe)
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(settings))
    return command, pipe


def test_guardian_death_requests_child_checkpoint(tmp_path):
    command, _ = blocked_fixture(tmp_path)
    guardian = subprocess.Popen(
        [sys.executable, "-c", "print('READY',flush=True);input()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert guardian.stdout.readline().strip() == "READY"
    command[-1] = str(guardian.pid)
    owner = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        while "READY" not in owner.stdout.readline():
            assert owner.poll() is None
        guardian.terminate()
        guardian.wait(timeout=10)
        stdout, stderr = owner.communicate(timeout=10)
        assert owner.returncode != 0 and "Accounting stopped" in stderr
        assert (
            json.loads((tmp_path / "output/training/result.json").read_text())["status"]
            == "budget_stopped"
        )
        assert (tmp_path / "executions").read_text() == "run\n"
    finally:
        for process in (owner, guardian):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def test_handle_setup_failure_drains_its_spawned_child(tmp_path, monkeypatch):
    task_fixture(tmp_path)
    opened = task_runner.open_process
    calls, spawned = [], []
    popen = subprocess.Popen

    def open_handle(pid):
        calls.append(pid)
        if len(calls) == 2:
            raise OSError("injected process-handle setup failure")
        return opened(pid)

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        spawned.append(child)
        return child

    monkeypatch.setattr(task_runner, "open_process", open_handle)
    monkeypatch.setattr(task_runner.subprocess, "Popen", spawn)
    import pytest

    with pytest.raises(OSError, match="injected"):
        task_runner.run(tmp_path / "task.yaml", tmp_path / "manifest.json", os.getpid())
    assert len(spawned) == 1 and spawned[0].returncode is not None


def test_completion_rejects_changed_epochs_and_nested_optimizer(tmp_path):
    task_fixture(tmp_path)
    output = tmp_path / "output"
    (output / "training").mkdir(parents=True)
    (output / "training/result.json").write_text(json.dumps({"status": "complete"}))
    expected = {"seed": 241, "epochs": 100, "optimizer": {"lr": 0.00024}}
    import pytest

    for saved in ({**expected, "epochs": 1}, {**expected, "optimizer": {"lr": 999}}):
        (output / "config.yaml").write_text(yaml.safe_dump(saved))
        with pytest.raises(ValueError, match="mismatched"):
            task_runner.check_complete(output, expected)


def prepared_fixture(tmp_path, monkeypatch):
    repository = SCRIPTS.parents[1]
    legacy = tmp_path / "legacy"
    configs = legacy / "source/configs"
    configs.mkdir(parents=True)
    for path in (repository / "configs").glob("*.yaml"):
        settings = yaml.safe_load(path.read_text())
        settings["output"] = str(tmp_path / "outputs" / path.stem)
        (configs / path.name).write_text(yaml.safe_dump(settings))
    (legacy / "software-contract.json").write_text("{}")
    (tmp_path / ".runs").mkdir()
    (tmp_path / ".runs/current-paper-launch.json").write_text(json.dumps({"root": str(legacy)}))
    executable = tmp_path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    import shlex

    executable.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    executable.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    return prepare()


def test_both_fixed_queues_wait_for_successful_source_finalization(tmp_path, monkeypatch):
    root = prepared_fixture(tmp_path, monkeypatch)
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["tasks"]) == len({t["output"] for t in manifest["tasks"]}) == 81
    assert sum(t["gpu"] == 0 for t in manifest["tasks"]) == 33
    fake = root / "runtime/task.py"
    fake.write_text(
        "import os,sys,pathlib,json,yaml\n"
        "p=pathlib.Path(sys.argv[sys.argv.index('--config')+1]);s=yaml.safe_load(p.read_text())\n"
        "with open(os.environ['EXECUTIONS'],'a') as f: f.write(p.stem+'\\n')\n"
        "if p.stem.startswith('tuab-source--'):\n"
        " result=pathlib.Path(s['output'])/'training/result.json';result.parent.mkdir(parents=True,exist_ok=True)\n"
        " result.write_text(json.dumps({'status':'complete'}))\n"
        " sys.exit(int(os.environ['SOURCE_FAIL']))\n"
    )
    for fail in (True, False):
        marker = root / "source-complete"
        marker.unlink(missing_ok=True)
        executions = tmp_path / f"executions-{fail}"
        env = {
            **os.environ,
            "EEGX_GUARDIAN": str(os.getpid()),
            "EXECUTIONS": str(executions),
            "SOURCE_FAIL": str(int(fail)),
        }
        with (root / "source.lock").open("a") as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)
            one = subprocess.Popen(["bash", str(root / "queue-gpu1.sh")], env=env)
            zero = subprocess.Popen(
                ["bash", str(root / "queue-gpu0.sh")],
                env={**env, "EEGX_SOURCE_FD": str(gate.fileno())},
                pass_fds=(gate.fileno(),),
            )
        assert zero.wait(timeout=30) == int(fail)
        assert one.wait(timeout=30) == int(fail)
        rows = executions.read_text().splitlines()
        cross = [
            r
            for r in rows
            if any(
                r.startswith(t)
                for t in ("crowdsourced-", "bciciv-2a-", "kalunga-", "physionet-mi-")
            )
        ]
        assert len(cross) == (0 if fail else 40)
        if not fail:
            assert len(rows) == len(set(rows)) == 81


def test_guardian_stops_at_aggregate_cap_and_drains_meter(tmp_path):
    repository = SCRIPTS.parents[1]
    primary, meter = tmp_path / "primary", tmp_path / "meter"
    Budget(primary, gpu_hours=3 / 3600, setup_seconds=2.7).initialize()
    Budget(meter, gpu_hours=3 / 3600).initialize()
    checkpoint = tmp_path / "safe-stop"
    actor = tmp_path / "actor.py"
    actor.write_text(
        "import sys,select,pathlib\n"
        f"sys.path.insert(0,{str(repository)!r})\n"
        "from pretrain.pipeline.budget import Budget,BudgetStop\n"
        f"b=Budget({str(meter)!r},gpu_hours=3/3600,max_concurrent=4)\n"
        "try:\n"
        " with b.stop_on_signals(),b.lease():\n"
        "  print('READY',flush=True)\n"
        "  while True: b.check();select.select([],[],[],0.01)\n"
        "except BudgetStop:\n"
        f" pathlib.Path({str(checkpoint)!r}).write_text(str(b.stop_requested))\n"
    )
    owner = subprocess.Popen([sys.executable, str(actor)], stdout=subprocess.PIPE, text=True)
    try:
        assert owner.stdout.readline().strip() == "READY"
        wait_actor = (
            f"import sys,select;sys.path.insert(0,{str(SCRIPTS)!r});"
            f"from process import open_process;select.select([open_process({owner.pid})],[],[])"
        )
        import shlex

        for device in (0, 1):
            (tmp_path / f"queue-gpu{device}.sh").write_text(
                f"#!/bin/sh\n{shlex.quote(sys.executable)} -c {shlex.quote(wait_actor)}\n"
            )
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "source_root": str(repository),
                    "budget": str(primary),
                    "meter": str(meter),
                    "claims": str(tmp_path / "claims"),
                }
            )
        )
        guardian = subprocess.run(
            [sys.executable, str(SCRIPTS / "launch.py"), "--manifest", str(manifest)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert guardian.returncode == 1, guardian.stderr
        owner.wait(timeout=10)
        assert checkpoint.read_text() == "True"
        state = json.loads((primary / "budget.json").read_text())
        meter_state = json.loads((meter / "budget.json").read_text())
        assert not state["active"] and not meter_state["active"]
        assert state["used_seconds"] >= state["limit_seconds"] == 3
        assert state["used_seconds"] == 2.7 + meter_state["used_seconds"]
        assert json.loads((tmp_path / "accounting.json").read_text())["active_gpu0"] == 0
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=10)


def test_wrapper_death_requests_training_checkpoint(tmp_path):
    import select
    import signal
    from process import open_process, signal_process

    command, _ = blocked_fixture(tmp_path)
    owner = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_handle = None
    try:
        while 'READY' not in owner.stdout.readline():
            assert owner.poll() is None
        receipt = next((tmp_path / 'claims').glob('*.json'))
        child_pid = json.loads(receipt.read_text())['pid']
        child_handle = open_process(child_pid)
        owner.kill()
        owner.wait(timeout=10)
        assert select.select([child_handle], [], [], 5)[0], 'Training outlived its killed wrapper'
        result = json.loads((tmp_path / 'output/training/result.json').read_text())
        assert result['status'] == 'budget_stopped'
    finally:
        if child_handle is not None:
            signal_process(child_handle, signal.SIGTERM)
            os.close(child_handle)
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=10)
        owner.stdout.close()
        owner.stderr.close()


def test_training_waits_for_durable_wrapper_receipt(tmp_path):
    import select
    import signal
    from process import open_process, signal_process

    command, _ = blocked_fixture(tmp_path)
    script = tmp_path / 'delayed_receipt.py'
    script.write_text(
        'import sys\n'
        f'sys.path.insert(0,{str(SCRIPTS)!r})\n'
        'import task\n'
        'original=task.save_receipt\n'
        'def delayed(path,value):\n'
        ' print("BEFORE_RECEIPT",value["pid"],flush=True)\n'
        ' sys.stdin.readline()\n'
        ' original(path,value)\n'
        'task.save_receipt=delayed\n'
        'task.main()\n')
    command[1] = str(script)
    owner = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    child_handle = None
    try:
        line = owner.stdout.readline().strip().split()
        assert line[0] == 'BEFORE_RECEIPT'
        child_handle = open_process(int(line[1]))
        select.select([owner.stdout], [], [], 1)
        assert not (tmp_path / 'executions').exists()
        owner.kill()
        owner.wait(timeout=10)
        assert select.select([child_handle], [], [], 5)[0]
        assert not (tmp_path / 'executions').exists()
    finally:
        if child_handle is not None:
            signal_process(child_handle, signal.SIGTERM)
            os.close(child_handle)
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=10)
        owner.stdin.close()
        owner.stdout.close()
        owner.stderr.close()
