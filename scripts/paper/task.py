"""Execute one explicit YAML task, with one lock per output directory."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import yaml

from process import open_process, process_start, open_matching_process, signal_process


def save_receipt(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def check_complete(output, settings):
    result = output / "training/result.json"
    if not result.is_file() or json.loads(result.read_text()).get("status") != "complete":
        raise RuntimeError(f"Task is incomplete; preserve its checkpoints: {output}")
    saved = yaml.safe_load((output / "config.yaml").read_text())

    def compare(expected, actual, prefix=""):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise ValueError(f"Completed task has mismatched {prefix}: {output}")
            for key, value in expected.items():
                if key not in actual:
                    raise ValueError(f"Completed task is missing {prefix}{key}: {output}")
                compare(value, actual[key], prefix + str(key) + ".")
        elif expected != actual:
            raise ValueError(f"Completed task has mismatched {prefix.rstrip('.')}: {output}")

    # The saved file includes resolved defaults. Compare every supplied setting,
    # including nested optimizer, epoch horizon, source and tracking identity.
    compare(settings, saved)
    if settings.get("purpose") == "paper":
        if not (output / "training/best.pth").is_file():
            raise RuntimeError(f"Selected checkpoint is missing: {output}")
        if settings["stage"] == "pretrain" and settings.get("probe_interval", 0):
            diagnostic = output / "training/probes/selected/diagnostics.json"
            if not diagnostic.is_file() or not json.loads(diagnostic.read_text()):
                raise RuntimeError(f"Selected-checkpoint probe is incomplete: {output}")


def verify_finalization(output, settings):
    """A foreign process has no wait status; confirm its tracking context closed."""
    if not settings.get("wandb", {}).get("enabled", False):
        return
    import time
    import threading
    import wandb

    receipt = json.loads((output / "wandb.json").read_text())
    deadline = time.monotonic() + 120
    while True:
        run = wandb.Api(timeout=30).run(f"{receipt['entity']}/{receipt['project']}/{receipt['id']}")
        if run.state == "finished" and run.summary.get("run_status") == "complete":
            return
        if run.state in {"failed", "crashed", "killed"} or time.monotonic() >= deadline:
            raise RuntimeError(f"W&B has not confirmed successful finalization: {run.id}")
        threading.Event().wait(1)


def run(config, manifest_path, guardian):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("adopt") or manifest.get("legacy_root"):
        raise RuntimeError("Live process handoff is disabled; use checkpoint resume")
    settings = yaml.safe_load(Path(config).read_text())
    output = Path(settings["output"])
    identity = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    claims = Path(manifest["claims"])
    claims.mkdir(parents=True, exist_ok=True)
    with (claims / (identity + ".lock")).open("a") as lock:
        # Do not queue the same logical task behind an active owner.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guardian_fd = open_process(guardian)
        pid = birth = child_fd = child = None
        complete = False
        stop_requested = False

        def stop(*_):
            nonlocal stop_requested
            stop_requested = True
            handle = child_fd
            if handle is not None:
                signal_process(handle, signal.SIGTERM)
            elif pid is not None:
                handle = open_matching_process(pid, birth)
                if handle is not None:
                    try:
                        signal_process(handle, signal.SIGTERM)
                    finally:
                        os.close(handle)

        previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            if select.select([guardian_fd], [], [], 0)[0]:
                raise RuntimeError("Shared GPU accounting process is not running")
            if output.exists():
                check_complete(output, settings)
                verify_finalization(output, settings)
                save_receipt(
                    claims / (identity + ".json"),
                    {
                        "output": str(output),
                        "pid": None,
                        "process_start": None,
                        "mode": "completed",
                        "status": "complete",
                        "device": settings["device"],
                    },
                )
                print(f"SKIP completed {output}", flush=True)
                return
            else:
                budget = json.loads((Path(manifest["budget"]) / "budget.json").read_text())
                if budget["used_seconds"] >= budget["limit_seconds"]:
                    raise RuntimeError("The shared GPU-hour budget is exhausted")
                health = Path(manifest_path).parent / "accounting.json"
                if stop_requested or (
                    health.exists() and json.loads(health.read_text())["stop_requested"]
                ):
                    raise RuntimeError("The shared accounting process requested a stop")
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(manifest["source_root"]) / "Main.py"),
                        "--config",
                        str(Path(config).resolve()),
                    ],
                    cwd=manifest["source_root"],
                )
                pid, birth = child.pid, process_start(child.pid)
                mode = "started"
            # Ownership starts at spawn, before handle or receipt setup.
            # All exceptions below request a checkpoint and drain the owned job.
            if child_fd is None:
                child_fd = open_process(pid)
            receipt = {
                "output": str(output),
                "pid": pid,
                "process_start": birth,
                "mode": mode,
                "device": settings["device"],
                "status": "running",
            }
            save_receipt(claims / (identity + ".json"), receipt)
            print(f"{mode.upper()} pid={pid} {output}", flush=True)
            if stop_requested:
                stop()
            ready = select.select([child_fd, guardian_fd], [], [])[0]
            if guardian_fd in ready and child_fd not in ready:
                raise RuntimeError("Accounting stopped; training was asked to checkpoint")
            if child is not None and child.wait() != 0:
                raise RuntimeError(f"Training failed; preserve its checkpoints: {output}")
            check_complete(output, settings)
            receipt["status"] = "complete"
            save_receipt(claims / (identity + ".json"), receipt)
            complete = True
            print(f"COMPLETE {output}", flush=True)
        finally:
            if pid is not None and not complete:
                stop()
                if child is not None:
                    child.wait()
                elif child_fd is not None:
                    select.select([child_fd], [], [])
            if child_fd is not None:
                os.close(child_fd)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            os.close(guardian_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--guardian", required=True, type=int)
    args = parser.parse_args()
    run(args.config, args.manifest, args.guardian)


if __name__ == "__main__":
    main()
