# Dictionary phase comparison

Fresh measurements use the same TUAB inputs, masks, backbone, precision and optimizer.
The GPU was idle at the start and after the benchmark exited.

| Setup | Value |
|---|---:|
| Microbatch / global batch | 512 / 4096 |
| Microbatches per update | 8 |
| Warmup / measured updates per method | 1 / 2 |
| EEG channels / samples | 19 / 1280 |

All methods ran in the same session with alternating measurement order.
Data loading is excluded.

| Measurement | Hard-count diagnostic | Bounded SoftDiCT Torch | SoftDiCT first fused | Optimized SoftDiCT | HYDRA dense |
|---|---:|---:|---:|---:|---:|
| Forward per microbatch, seconds | 1.498 | 0.790 | 0.263 | 0.157 | 0.187 |
| Backward per microbatch, seconds | 0.114 | 1.062 | 0.724 | 0.228 | 0.114 |
| Gradient clipping per update, milliseconds | 3.23 | 2.86 | 2.66 | 3.01 | 2.63 |
| Optimizer step per update, milliseconds | 8.22 | 7.59 | 7.24 | 7.49 | 7.48 |
| Full update, seconds | 12.917 | 14.829 | 7.926 | 3.102 | 2.427 |
| Peak process allocation, GiB | 11.26 | 11.86 | 12.30 | 10.86 | 11.14 |
| Auxiliary reconstruction gradient | Absent | Present | Present | Present | Through target head |

Backward remains the largest phase for SoftDiCT; optimizer steps contribute little.
This native hard-count implementation spends most of its time in forward computation.

Hard DiCT is a diagnostic: its feature loss has no autograd path, so backward uses only raw reconstruction loss.
Optimized SoftDiCT preserves the agreed objective and full dictionary; HYDRA dense changes both the target objective and dictionary geometry.

The bounded PyTorch column already uses memory and convolution optimizations. Main-branch DiCT instead uses a straight-through estimator and response-weighted features; it is separate from the hard-count diagnostic above.
