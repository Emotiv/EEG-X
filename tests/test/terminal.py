"""A live study in a real tmux terminal must never be detached by the launcher."""

import json
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

import pytest

from pretrain.pipeline.budget import Budget


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY / "scripts/paper"
sys.path.insert(0, str(SCRIPTS))
from process import open_matching_process, process_start, signal_process  # noqa: E402


FAKE_MAIN = r"""
import json, os, pathlib, select, signal, socket, subprocess, sys
root = pathlib.Path(sys.argv[-1]).parent.parent
sys.path.insert(0, REPOSITORY)
from pretrain.pipeline.budget import Budget
settings = json.loads(pathlib.Path(sys.argv[-1]).read_text())
if settings.get("parent"):
    subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()),
                    "--config", settings["child"]], check=True)
else:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    stopping = [False]
    terminal = os.open("/dev/tty", os.O_RDONLY)
    tty = os.ttyname(terminal)
    os.close(terminal)
    def event(kind, number=None):
        message = {"kind": kind, "signal": number, "pid": os.getpid(),
                   "pgrp": os.getpgrp(), "sid": os.getsid(0), "tty": tty}
        connection.sendto(json.dumps(message).encode(), str(root / "events.sock"))
    def on_signal(number, _):
        event("signal", signal.Signals(number).name)
        if number in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            stopping[0] = True
    for number in (signal.SIGHUP, signal.SIGCONT, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, on_signal)
    output = pathlib.Path(settings["output"])
    output.mkdir()
    with Budget(root / "budget", max_concurrent=4, device="GPU-fake").lease():
        step = 0
        (output / "step").write_text("0")
        event("ready")
        while not stopping[0]:
            step += 1
            (output / "step.partial").write_text(str(step))
            os.replace(output / "step.partial", output / "step")
            select.select([], [], [], 0.02)
"""


@pytest.mark.skipif(shutil.which("tmux") is None, reason="Requires a real tmux terminal")
def test_live_tmux_study_is_rejected_without_signals_or_dispatcher_changes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Main.py").write_text(FAKE_MAIN.replace("REPOSITORY", repr(str(REPOSITORY))))
    Budget(tmp_path / "budget", max_concurrent=4).initialize()
    Budget(tmp_path / "meter", max_concurrent=4).initialize()
    session = "eegx-terminal-test-" + uuid.uuid4().hex[:12]
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(tmp_path / "events.sock"))
    server.settimeout(10)
    quote = shlex.quote
    shell = [
        "#!/usr/bin/env bash",
        "set -e",
        f"python_bin={quote(sys.executable)}",
        'pipeline() { "$python_bin" Main.py --config "$1"; }',
    ]
    tasks = []
    for number in range(4):
        output = str(tmp_path / f"output-{number}")
        child = source / f"child-{number}.json"
        child.write_text(json.dumps({"seed": 241, "output": output}))
        parent = source / f"parent-{number}.json"
        parent.write_text(json.dumps({"parent": True, "child": str(child)}))
        shell.append(
            f"pipeline {quote(str(parent))} >{quote(str(tmp_path / f'pipeline-{number}.log'))} 2>&1 &"
        )
        tasks.append({"output": output, "gpu": 1})
    (source / "run_paper.sh").write_text("\n".join([*shell, "wait"]) + "\n")
    (tmp_path / "launch.sh").write_text(
        f"#!/usr/bin/env bash\ncd {quote(str(source))}\n"
        f"bash run_paper.sh >{quote(str(tmp_path / 'launcher.log'))} 2>&1\n"
    )
    for device in (0, 1):
        (tmp_path / f"queue-gpu{device}.sh").write_text(
            f"#!/usr/bin/env bash\ntouch {quote(str(tmp_path / 'unexpected-dispatch'))}\n"
        )
    owners = []
    try:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "-c",
                str(source),
                "bash",
                str(tmp_path / "launch.sh"),
            ],
            check=True,
        )
        for _ in range(4):
            ready = json.loads(server.recv(65536))
            assert ready["kind"] == "ready" and ready["tty"] == "/dev/tty"
            owners.append({"pid": ready["pid"], "process_start": process_start(ready["pid"])})
        first_steps = [int((Path(task["output"]) / "step").read_text()) for task in tasks]
        manifest = {
            "legacy_root": str(tmp_path),
            "source_root": str(source),
            "python": sys.executable,
            "budget": str(tmp_path / "budget"),
            "meter": str(tmp_path / "meter"),
            "claims": str(tmp_path / "claims"),
            "tasks": tasks,
            "adopt": {},
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        rejected = subprocess.run(
            [sys.executable, str(SCRIPTS / "launch.py"), "--manifest", str(path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert rejected.returncode != 0, rejected.stdout + rejected.stderr
        assert "live" in rejected.stderr.lower(), rejected.stderr
        assert not (tmp_path / "handoff.json").exists()
        assert not (tmp_path / "unexpected-dispatch").exists()
        assert all(process_start(p["pid"]) == p["process_start"] for p in owners)
        # No SIGCONT or SIGHUP may be delivered. Waiting on the event socket also
        # gives the heartbeat counters time to prove that each child kept running.
        assert not select.select([server], [], [], 0.15)[0]
        last_steps = [int((Path(task["output"]) / "step").read_text()) for task in tasks]
        assert all(last > first for first, last in zip(first_steps, last_steps))
    finally:
        descriptors = []
        # Include children that acquired a lease but whose readiness event was
        # not consumed if fixture setup or an assertion failed.
        ledger = json.loads((tmp_path / "budget/budget.json").read_text())
        owners.extend(ledger.get("leases", {}).values())
        seen = set()
        for owner in owners:
            if owner["pid"] in seen:
                continue
            seen.add(owner["pid"])
            handle = open_matching_process(owner["pid"], owner["process_start"])
            if handle is not None:
                signal_process(handle, signal.SIGTERM)
                descriptors.append(handle)
        deadline = time.monotonic() + 5
        while descriptors and time.monotonic() < deadline:
            for descriptor in select.select(descriptors, [], [], 0.1)[0]:
                descriptors.remove(descriptor)
                os.close(descriptor)
        for descriptor in descriptors:
            os.close(descriptor)
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        server.close()
