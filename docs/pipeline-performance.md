# Recorded EEG-X speed check

These measurements precede the selected XYZ and Soft DiCT default update. They
describe the earlier implementation; the new defaults have not been timed on GPU.

Real EEG training completed on the user-selected GPU while other workloads remained active. Both tasks used forward, backward, and optimizer updates.

| Setting | Value |
|---|---|
| Physical GPU | 0 |
| Device | NVIDIA RTX PRO 6000 Blackwell Max-Q |
| Encoder depth / width / heads | 8 / 64 / 8 |
| Objective | Raw masked MSE plus Soft Hydra |
| Precision | BF16 |
| CPU threads | 4 |
| Logical / physical batch | 4 / 4 |
| Warmup / measured updates per task | 5 / 20 |
| PyTorch allocator limit | 8 GiB |
| Compilation | Off |

An update is one parameter change after forward and backward computation. Timings include batch loading and optimizer work, with no gradient accumulation.

| Dataset | Window | Channels | Windows/s | Median update, s | Peak PyTorch allocation, GiB | Attention |
|---|---:|---:|---:|---:|---:|---|
| DREAMER arousal | 2 s | 14 | 34.06 | 0.117 | 0.39 | FlashAttention |
| TUAB | 10 s | 19 | 20.06 | 0.199 | 2.96 | FlashAttention |

Both tasks completed their bounded training checks. Peak allocation covers PyTorch tensors, not total device memory.

| Contention and budget | DREAMER | TUAB |
|---|---:|---:|
| Device utilization before / after | 99% / 97% | 99% / 97% |
| Charged GPU time, s | 7.78 | 38.43 |

The existing budget ledger includes both runs. Other compute workloads were present before each run.

These measurements show throughput under contention. They cannot establish an isolated GPU speed or a code speedup. Previous EEGX2 runs use a different model and cannot provide a matched comparison. These short training runs cannot supply paper accuracy results.
