# Original versus current EEG-X training cost

This comparison uses the original EEG-X implementation from Git and the current
EEG-X model. The encoder dimensions match. The tokenizer, masking, decoder,
normalization and teacher computation differ.

| Measurement condition | Both models |
|---|---|
| Encoder depth / width / heads | 8 / 64 / 8 |
| Logical / physical batch | 4 / 4; no gradient accumulation |
| Precision / CPU threads / loader workers | BF16 / 4 / 2 |
| Device | Physical GPU 0, NVIDIA RTX PRO 6000 Blackwell Max-Q; shared |
| Allocator limit / compilation | 8 GiB / off |
| Repeats per dataset | 3 alternating pairs; 5 warmup and 20 measured updates per run |
| Loss / optimizer | Full-waveform raw MSE against the same input / AdamW |

An update includes batch loading, transfer, forward and backward computation,
gradient clipping, optimizer work and the original teacher update where present.
Startup and input checks are excluded from timing; GPU work uses the existing budget.

| Dataset; channels × samples | Before windows/s | After windows/s | Before update, ms | After update, ms | Observed throughput change |
|---|---:|---:|---:|---:|---:|
| DREAMER arousal; 14 × 256 | 72.82 | 94.60 | 54.93 | 42.28 | +29.9% |
| TUAB; 19 × 1280 | 70.56 | 92.48 | 56.69 | 43.25 | +31.1% |

Results use the median across repeated runs. These are observed costs on a shared
GPU; contention and model changes prevent a code-speedup conclusion.

| Dataset | Before allocated, GiB | After allocated, GiB | Allocation change | Before reserved, GiB | After reserved, GiB |
|---|---:|---:|---:|---:|---:|
| DREAMER arousal | 0.080 | 0.084 | +5.3% | 0.082 | 0.086 |
| TUAB | 0.478 | 0.152 | -68.1% | 0.605 | 0.182 |

Peak allocation measures this process's PyTorch tensors. Reserved memory includes
the allocator cache; neither measure represents total device memory.

| Model difference | Original EEG-X | Current EEG-X |
|---|---|---|
| Patch length / stride, samples | 128 / 32 | 32 / 32 |
| Masking | Subsequence sampling; requested 50% | Fixed 75% per window |
| Total parameters, DREAMER | 912,643 | 653,626 |
| Total parameters, TUAB | 912,653 | 653,626 |
| Predictor / decoder | 2 blocks / transposed convolution | 1 block / 4 Transformer blocks |
| Teacher | Full target encoder and momentum update | None |
| Input / normalization / attention | Spectral / LayerNorm / manual | Raw plus spectral / RMSNorm with Q/K normalization / FlashAttention |

The original release imports a missing dictionary implementation. This diagnostic
uses a common raw MSE loss and optimizer, with dictionary losses disabled for both
models. It does not reproduce either complete training recipe or establish paper accuracy.

[Recorded measurements](measurements/eegx-performance.json) preserve update times, model settings, source digests and device load.
