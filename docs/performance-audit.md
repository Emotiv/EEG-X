# Training performance audit

The main runtime limit is GPU dictionary work. SoftDiCT converts each signal
into fixed-filter probability counts. Large temporary tensors and many small
kernel launches made this step expensive; input loading was much faster.

| Input check | Measurement |
|---|---:|
| TUAB loader batch | 256 windows |
| Mean loader wait | 0.066 s |
| Mean mask construction | 0.0064 s |
| Existing profiled optimizer update | 5.56 s |
| Attention backend | FlashAttention |

A busy CPU thread includes waiting for CUDA. CPU usage alone does not show a
CPU computation limit.

| GPU benchmark | Bounded Torch baseline | Fused GPU candidate |
|---|---:|---:|
| Microbatch / global batch | 512 / 4096 | 512 / 4096 |
| Warmup / timed updates | 1 / 2 | 1 / 2 |
| Median update time, without profiler | 39.58 s | 25.22 s |
| Peak PyTorch allocation | 11.79 GiB | 12.26 GiB |
| Profiled GPU kernel launches per update | 105,926 | 29,430 |
| Observed update-time reduction | Reference | 36.3% |

Both cases used the same input, seed and batches on a shared GPU. These short
measurements show observed improvement, not an isolated speedup or full-study estimate.

| Validation | Result |
|---|---:|
| Maximum GPU feature error against original convolution | 7.63e-6 |
| Maximum GPU input-gradient error | 7.45e-9 |
| GPU parity cases, including temporal tails and mixed precision | 5 passed |
| GPU random-state preservation | Passed |
| Full CPU suite | 223 passed; 4 skipped |
| Standards / Spec review | 0 open / 0 open |

The fused candidate preserves the dictionary math within floating-point tolerances.
Torch remains the default; the optional GPU backend supports first-order gradients.

The slower generic compiler candidate was rejected. This work is prepared and
benchmarked only. Paper jobs, their batch settings and W&B histories remain unchanged.
