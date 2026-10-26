"""Persistent GPU wall-time budget shared by bounded concurrent runs on one device."""

import contextlib
import fcntl
import json
import math
import os
import signal
import time
import uuid
from pathlib import Path

from pretrain.metadata import atomic_json


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, root, *, gpu_hours=72, setup_seconds=0, max_concurrent=1, device=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "budget.json"
        self.limit = float(gpu_hours) * 3600
        if not math.isfinite(self.limit) or self.limit <= 0 or self.limit > 72 * 3600:
            raise ValueError("Study budget must be positive and cannot exceed the accepted cap")
        self.setup_seconds = float(setup_seconds)
        if not math.isfinite(self.setup_seconds) or self.setup_seconds < 0:
            raise ValueError("Setup allowance must be finite and nonnegative")
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        self.max_concurrent = max_concurrent
        self.device = device or os.environ.get("CUDA_VISIBLE_DEVICES", "unspecified")
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self.token = uuid.uuid4().hex
        self.started = None
        self.stop_requested = False

    @contextlib.contextmanager
    def _locked(self):
        with (self.root / "budget.lock").open("a") as lock:
            # Lock only ledger transactions. Training processes keep separate leases.
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    @staticmethod
    def _process(pid):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return None if fields[0] == "Z" else fields[19]
        except FileNotFoundError:
            return None

    def _settle(self, state):
        """Charge an occupied GPU once, including conservative crash recovery."""
        now, monotonic = time.time(), time.perf_counter()
        if state["active"]:
            delta = (
                monotonic - state["updated_monotonic"]
                if state.get("boot_id") == self.boot_id and "updated_monotonic" in state
                else now - state["updated_utc"]
            )
            state["used_seconds"] += max(0, delta)
        state["leases"] = {
            token: lease
            for token, lease in state.get("leases", {}).items()
            if lease["boot_id"] == self.boot_id
            and self._process(lease["pid"]) == lease["process_start"]
        }
        state.update(
            active=bool(state["leases"]),
            updated_utc=now,
            updated_monotonic=monotonic,
            boot_id=self.boot_id,
        )

    def _read(self):
        state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "used_seconds": self.setup_seconds,
                "setup_seconds": self.setup_seconds,
                "limit_seconds": self.limit,
                "active": False,
                "updated_utc": time.time(),
            }
        )
        if state["limit_seconds"] != self.limit:
            raise ValueError("Budget changed; automatic extension is forbidden")
        return state

    def initialize(self):
        """Charge setup once for new ledgers; preserve previously charged work."""
        with self._locked():
            state = self._read()
            if not self.path.exists():
                atomic_json(self.path, state)
            return state

    @contextlib.contextmanager
    def stop_on_signals(self):
        """Request a safe boundary on interrupt and restore the caller's handlers."""
        previous = {}

        def request_stop(*_):
            self.stop_requested = True

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, request_stop)
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    @contextlib.contextmanager
    def lease(self):
        if self.started is not None:
            raise RuntimeError("Budget already has an active lease")
        with self._locked():
            state = self._read()
            self._settle(state)
            # Preserve recovered usage even if this admission is rejected.
            atomic_json(self.path, state)
            leases = state["leases"]
            if leases and (
                state["device"] != self.device or state["max_concurrent"] != self.max_concurrent
            ):
                raise ValueError("Concurrent runs must share one GPU and one concurrency limit")
            if len(leases) >= self.max_concurrent:
                raise BlockingIOError("The shared GPU concurrency limit is occupied")
            self.used = state["used_seconds"]
            if self.used >= self.limit:
                raise BudgetStop("The shared GPU-hour budget is exhausted")
            self.started = time.perf_counter()
            leases[self.token] = {
                "pid": os.getpid(),
                "process_start": self._process(os.getpid()),
                "boot_id": self.boot_id,
            }
            state.update(active=True, device=self.device, max_concurrent=self.max_concurrent)
            atomic_json(self.path, state)
        try:
            yield self
        finally:
            try:
                self.flush(active=False)
            finally:
                self.used = self.elapsed
                self.started = None

    @property
    def elapsed(self):
        return self.used + (time.perf_counter() - self.started if self.started is not None else 0)

    def check(self):
        if self.started is None:
            raise RuntimeError("GPU work requires an active budget lease")
        if self.stop_requested or self.elapsed >= self.limit:
            raise BudgetStop("Stop requested or shared GPU-hour budget reached")

    def flush(self, *, active=True):
        if self.started is None:
            raise RuntimeError("GPU work requires an active budget lease")
        with self._locked():
            state = self._read()
            self._settle(state)
            if self.token not in state["leases"]:
                raise RuntimeError("Active GPU lease disappeared")
            if not active:
                del state["leases"][self.token]
            state["active"] = bool(state["leases"])
            atomic_json(self.path, state)


class NoGpuBudget:
    """CPU integration checks do not spend GPU time."""

    def check(self):
        pass

    def flush(self):
        pass
