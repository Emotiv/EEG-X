# GPU allocation-fill optimization

This engineering option skips allocation fill while keeping deterministic algorithms enabled. It preserves the selected SoftDiCT objective in the measured checks.

| Measurement | Result |
|---|---:|
| Microbatch / global batch | 512 / 4096 |
| Original model update | 2.981 s |
| Candidate model update | 2.823 s |
| Update-time reduction | 5.3% |
| Warmup / measured updates per variant | 1 / 2 |
| Peak GPU allocation, both variants | 10.83 GiB |
| Model parameter and gradient tensors compared | 216 |
| Dictionary cases / seeds | 13 / 5 |
| CPU regression checks | 70 passed, 2 GPU-only skipped |

Paired updates reuse the same real EEG batch and masks. Losses match on every update; final parameters, gradients and RNG states match bitwise.

| Policy | Result |
|---|---|
| Deterministic algorithms | Enabled |
| Allocation-fill option | Engineering benchmark only |
| Checkpoint and resume | Choice recorded; matching-policy resume and mismatched-policy rejection passed |
| State restoration | Verified after nested scopes and errors |
| Paper training configuration | Unchanged |

This option disables a process-wide diagnostic that can expose uninitialized-memory reads. Full-epoch GPU performance and all possible shapes have not been measured.
