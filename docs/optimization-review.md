# Optimization review

| Review axis | Open findings | Result |
|---|---:|---|
| Standards | 0 | Report formatting fixed |
| Specification | 0 | Matches the selected batch, backend and determinism policy |
| CPU regression | 256 passed, 6 skipped | End-to-end training, transfer, batching and resume checks passed |
| GPU resume | 3 updates, interruption after 1 | Model, optimizer, random state and contract match bitwise |
| Validation inference | 8 datasets, 16 samples each | Maximum encoder difference 2.384e-7; identical initialized-head logits |
| Loader checks | 17 configurations | No worker shutdown warnings after cleanup fix |

The review found no confirmed code defect. The checkpoint documentation now states that backend transfer requires the same software and environment contract.

| Foreseeable issue | Implication before full training |
|---|---|
| Larger global batch | Fewer optimizer updates can change final accuracy; complete the agreed seed study |
| Full-epoch GPU coverage | Validation, scheduled probes and final checkpoint selection still need an integrated GPU run |
| Concurrent GPU coverage | Simultaneous admission of 4 streams has not been measured under this recipe |
| STEW fused normalization | Short measurements were 1.2%-8.5% slower; keep this visible when comparing throughput |
| Uninitialized-memory diagnostic disabled | Deterministic algorithms remain enabled, but unwritten-buffer bugs can be hidden |
| Checkpoint contracts | Fresh training is required for the changed recipe; historical-source checkpoints remain unsupported |
| Original paper budget | Exhausted; these checks do not authorize or start full paper training |

These are known validation limits and execution choices, rather than confirmed defects. Short numerical comparisons do not establish final paper accuracy.
