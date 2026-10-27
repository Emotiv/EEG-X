# GPU training profile

The prepared optimized Triton training path has substantial gains and still has GPU optimization opportunities. This audit covers real HDF5 training updates.

| Measurement | Result |
|---|---:|
| Profiled microbatch / global batch | 512 / 4096 |
| Full training update, profiler disabled | 3.152 s |
| Instrumented update | 3.283 s |
| Peak PyTorch GPU allocation | 10.82 GiB |
| Device kernel/copy activity within instrumented interval | 89.5% |
| Actual GPU kernel calls per instrumented update | 29,110 |
| Attention backend | FlashAttention throughout |
| Paper YAML backend, micro / global | Torch, 16 / 256 |

Device activity is a trace interval measure, not hardware occupancy. The paper YAMLs have not adopted the benchmark recipe.

| GPU kernel work per instrumented update | Time |
|---|---:|
| Dictionary forward | 0.646 s |
| Dictionary backward and fold | 0.830 s |
| FP32 RMSNorm forward and backward | 0.433 s |
| FlashAttention kernels | 0.294 s |
| Buffer fill kernels | 0.169 s |
| Identified copy/cast kernels | 0.120 s |

Dictionary kernels remain the largest GPU cost. These are baseline costs before the later normalization and allocation improvements.

Most fill operations occur inside allocation calls while deterministic uninitialized-memory filling is enabled. No kernel changes were made during this audit.

The forced profiling stop emitted a native cleanup diagnostic; the parent exited successfully and checkpoint checks passed. Validation, embedded probes and online tracking were outside this short profile.
