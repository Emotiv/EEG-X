"""Account for a second GPU without changing an already-running training snapshot."""

import json
import os
import signal
from pathlib import Path

from pretrain.metadata import atomic_json
from pretrain.pipeline.budget import Budget
from process import open_matching_process, signal_process


class SharedAccounting:
    """The primary ledger owns the cap; the auxiliary ledger meters GPU zero only."""

    def __init__(self, primary, meter):
        primary, meter = Path(primary), Path(meter)
        hours = json.loads((primary / "budget.json").read_text())["limit_seconds"] / 3600
        self.primary = Budget(primary, gpu_hours=hours)
        self.meter = Budget(meter, gpu_hours=hours)
        self.key = str(meter.resolve())

    def settle(self):
        # Both legacy writers take just their own lock. This lock order cannot
        # deadlock with them, and a legacy flush preserves the extra debit fields.
        with self.primary._locked(), self.meter._locked():
            primary, meter = self.primary._read(), self.meter._read()
            self.primary._settle(primary)
            self.meter._settle(meter)
            debits = primary.setdefault("external_meters", {})
            previous = debits.get(self.key, 0.0)
            if meter["used_seconds"] < previous:
                raise ValueError("Auxiliary GPU usage decreased; refusing to reset the meter")
            primary["used_seconds"] += meter["used_seconds"] - previous
            debits[self.key] = meter["used_seconds"]
            # If interrupted between writes, the next settlement charges the
            # unrecorded delta exactly once from the primary debit watermark.
            atomic_json(self.meter.path, meter)
            atomic_json(self.primary.path, primary)
            return primary, meter

    @staticmethod
    def request_stop(*states):
        """Signal only verified owners of these study ledgers, at safe boundaries."""
        for state in states:
            for lease in state.get("leases", {}).values():
                handle = open_matching_process(lease["pid"], lease["process_start"])
                if handle is not None:
                    try:
                        signal_process(handle, signal.SIGTERM)
                    finally:
                        os.close(handle)
