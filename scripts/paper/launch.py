"""Run two fixed paper queues under the original aggregate GPU-hour cap."""

import argparse
import fcntl
import json
import hashlib
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from process import open_process, process_start


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("legacy_root") or manifest.get("adopt"):
        raise RuntimeError("Live process handoff is disabled; use checkpoint resume")
    if set(manifest.get("gpu1_ready_outputs", [])) != set(manifest.get("expected_run_ids", {})):
        raise ValueError("Every resumed GPU-one task requires its original W&B identity")
    root = manifest_path.parent
    # Import the same frozen budget implementation as the active training jobs.
    sys.path.insert(0, manifest["source_root"])
    from accounting import SharedAccounting
    from pretrain.metadata import atomic_json

    contract = manifest.get("software_contract")
    if contract:
        paths = {
            **contract["sources"],
            "uv.lock": contract["uv_lock"],
            "docs/spec.md": contract["specification"],
        }
        for relative, expected in paths.items():
            value = hashlib.sha256(
                (Path(manifest["source_root"]) / relative).read_bytes()
            ).hexdigest()
            if value != expected:
                raise ValueError(f"Frozen source changed: {relative}")
    with (Path(manifest["budget"]) / "queues.lock").open("a") as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        accounting = SharedAccounting(manifest["budget"], manifest["meter"])
        first, _ = accounting.settle()
        if first["used_seconds"] >= first["limit_seconds"]:
            raise RuntimeError("Shared GPU-hour budget exhausted; no queue handoff performed")
        stopped = False
        queues, handles, descriptors = [], [], {}

        def stop(*_):
            nonlocal stopped
            stopped = True

        def settle():
            nonlocal stopped
            primary, meter = accounting.settle()
            stopped = stopped or primary["used_seconds"] >= primary["limit_seconds"]
            atomic_json(
                root / "accounting.json",
                {
                    "used_gpu_hours": primary["used_seconds"] / 3600,
                    "limit_gpu_hours": primary["limit_seconds"] / 3600,
                    "active_gpu1": len(primary["leases"]),
                    "active_gpu0": len(meter["leases"]),
                    "stop_requested": stopped,
                    "updated_utc": time.time(),
                },
            )
            if stopped:
                accounting.request_stop(primary, meter)
            return primary, meter

        def primary_ready():
            owners = accounting.primary._read().get("leases", {})
            live = {
                p["pid"] for p in owners.values() if process_start(p["pid"]) == p["process_start"]
            }
            for output in manifest.get("gpu1_ready_outputs", []):
                identity = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()
                path = Path(manifest["claims"]) / (identity + ".json")
                if not path.exists() or json.loads(path.read_text()).get("pid") not in live:
                    return False
                expected = manifest.get("expected_run_ids", {}).get(output)
                if expected:
                    tracking = Path(output) / "wandb.json"
                    if not tracking.exists():
                        return False
                    if json.loads(tracking.read_text())["id"] != expected:
                        raise ValueError(
                            "Resumed W&B run identity changed; GPU zero was not launched"
                        )
            return True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        env = {
            **os.environ,
            "EEGX_GUARDIAN": str(os.getpid()),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "PYTHONUNBUFFERED": "1",
        }
        codes = None
        try:
            with (root / "source.lock").open("a") as gate:
                # Own the source gate before starting either queue. GPU zero
                # inherits it, and releases it only after source finalization.
                fcntl.flock(gate, fcntl.LOCK_EX)
                for device in (1, 0):
                    if device == 0:
                        # GPU one establishes owned tasks and accounting-death watchers before any
                        # second-GPU work starts. Resume failures stop before GPU zero admission.
                        deadline = time.monotonic() + 120
                        while not primary_ready():
                            settle()
                            if (
                                stopped
                                or queues[0].poll() is not None
                                or time.monotonic() >= deadline
                            ):
                                raise RuntimeError(
                                    "GPU one did not establish its resumed leases; GPU zero was not launched"
                                )
                            select.select(list(descriptors), [], [], 0.1)
                    log = (root / f"queue-gpu{device}.log").open("a")
                    handles.append(log)
                    process = subprocess.Popen(
                        ["bash", str(root / f"queue-gpu{device}.sh")],
                        env={**env, "EEGX_SOURCE_FD": str(gate.fileno())},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        pass_fds=(gate.fileno(),) if device == 0 else (),
                        start_new_session=True,
                    )
                    queues.append(process)
                    descriptors[open_process(process.pid)] = process
            atomic_json(
                root / "guardian.json", {"pid": os.getpid(), "queues": [p.pid for p in queues]}
            )
            while descriptors:
                settle()
                for fd in select.select(list(descriptors), [], [], 1)[0]:
                    process = descriptors.pop(fd)
                    process.wait()
                    os.close(fd)
                    if process.returncode:
                        stopped = True
            codes = [p.returncode for p in queues]
        finally:
            # Include owners whose shell/helper already died. Keep metering until
            # every actual lease has drained, then write the final debit receipt.
            primary, meter = accounting.settle()
            if any(state["active"] for state in (primary, meter)):
                stopped = True
                accounting.request_stop(primary, meter)
                while any(state["active"] for state in (primary, meter)):
                    primary, meter = settle()
                    if any(state["active"] for state in (primary, meter)):
                        select.select([], [], [], 0.1)
            settle()
            for fd in descriptors:
                os.close(fd)
            for handle in handles:
                handle.close()
            atomic_json(
                root / "finished.json",
                {"exit_codes": codes, "stopped": stopped, "updated_utc": time.time()},
            )
        return int(stopped or any(codes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest")
    group.add_argument("--current")
    args = parser.parse_args()
    manifest = args.manifest
    if args.current:
        manifest = json.loads(Path(args.current).read_text())["manifest"]
    return run(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
