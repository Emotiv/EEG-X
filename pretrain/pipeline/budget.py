"""Persistent single-GPU budget with exclusive ownership and conservative recovery."""

import contextlib
import fcntl
import json
import math
import signal
import time
from pathlib import Path

from pretrain.metadata import atomic_json


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, root, *, gpu_hours=72, setup_seconds=0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "budget.json"
        self.limit = float(gpu_hours) * 3600
        if not math.isfinite(self.limit) or self.limit <= 0 or self.limit > 72 * 3600:
            raise ValueError("Study budget must be positive and cannot exceed the accepted cap")
        self.setup_seconds = float(setup_seconds)
        if not math.isfinite(self.setup_seconds) or self.setup_seconds < 0:
            raise ValueError("Setup allowance must be finite and nonnegative")
        self.started = None
        self.stop_requested = False

    @contextlib.contextmanager
    def _locked(self):
        with (self.root / "budget.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

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
        with self._locked():
            self.state = state = self._read()
            if state["active"]:
                # A vanished writer may have reserved the device since its last receipt.
                state["used_seconds"] += max(0, time.time() - state["updated_utc"])
            self.used = state["used_seconds"]
            if self.used >= self.limit:
                raise BudgetStop("The shared GPU-hour budget is exhausted")
            self.started = time.perf_counter()
            self.flush(active=True)
            try:
                yield self
            finally:
                self.flush(active=False)
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
        atomic_json(
            self.path,
            {
                **self.state,
                "used_seconds": self.elapsed,
                "limit_seconds": self.limit,
                "active": active,
                "updated_utc": time.time(),
            },
        )


class NoGpuBudget:
    """CPU integration checks do not spend GPU time."""

    def check(self):
        pass

    def flush(self):
        pass
