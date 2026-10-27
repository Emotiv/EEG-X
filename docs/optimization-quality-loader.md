# Optimization quality and data loading

| Check | Result |
|---|---|
| Paper configurations | 17 select fused normalization and disable diagnostic memory fill |
| Pretraining batch | 512 microbatch, 4096 global batch; fused SoftDiCT |
| Deterministic algorithms | Enabled |
| GPU checkpoint resume | 3 uninterrupted updates matched 1 plus 2 resumed updates bitwise for model, optimizer, RNG and state |
| Validation inference | 8 datasets, 16 samples each; maximum encoder difference 2.384e-7, identical logits |
| Inference checkpoint age | 3 training updates |
| Loader workers | 2 |
| Loader early exit | Worker shutdown clean; sample order and RNG match the native loader |

The inference comparison uses a short-run checkpoint and an initialized classifier. It does not measure final accuracy. Disabling diagnostic memory fill may hide bugs that read unwritten memory.

| TUAB GPU training measurement | Time |
|---|---:|
| First logical batch, including worker startup | 5.83 s |
| Later batch wait, median of 3 observations | 0.00027 s |
| Mask creation, later updates | 0.17-0.19 s |
| Warm training update | 2.60 s |

This short measurement uses real HDF5 data. It excludes validation and tracking; cache and shared-machine conditions can change the results.

| CPU loader check | First logical batch | Median of 3 later waits |
|---|---:|---:|
| DREAMER pretraining | 3.47 s | 0.011 s |
| STEW pretraining | 3.32 s | 0.012 s |
| TUAB pretraining | 5.43 s | 0.056 s |
| TUEV pretraining | 3.51 s | 0.020 s |
| Downstream stages | 2.34-2.56 s | 0.001-0.011 s |

These CPU-only checks cover all configured stages and use a small sample. Individual waits can be longer than the median.

| Quality or speed limitation | Evidence |
|---|---|
| Batch recipe change | TUAB optimizer updates per epoch change from 1137 to 72 |
| Final paper metrics | Not yet measured over the full 5-seed study |
| STEW fused normalization | 1.2%-8.5% slower in short model tests |
| Old training checkpoints | Different batch and compute contracts cannot resume exactly under the new recipe |
| Paper GPU ledger | Exhausted; no paper runs started by these checks |

The equations stay the same, but floating-point rounding and the changed batch can change training results. The user selected the fused backend for STEW despite its measured slowdown.
