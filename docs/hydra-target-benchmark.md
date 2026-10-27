# SoftDiCT and HYDRA target-head speed comparison

The HYDRA target head had the lowest measured microbatch time in these shared-GPU checks.
A microbatch and an accumulated optimizer update measure different amounts of work.

| Large-batch setup | Value |
|---|---:|
| Microbatch / global batch | 512 / 4096 |
| Microbatches per update | 8 |
| TUAB channels / samples per window | 19 / 1280 |
| Encoder depth / width / heads | 8 / 64 / 8 |
| Warmup / timed updates per case | 1 / 2 |

The large-batch cases used identical real EEG values and masks, plus the same backbone,
seed, precision and optimizer. Measurements ran sequentially on the same shared GPU.

| Direct model measurement | Bounded Torch SoftDiCT | Fused GPU SoftDiCT | HYDRA target head |
|---|---:|---:|---:|
| Forward | 2.53 s | 0.84 s | 0.62 s |
| Backward | 3.33 s | 2.29 s | 0.37 s |
| Complete microbatch | 5.86 s | 3.12 s | 0.98 s |
| Optimizer update | 46.86 s | 25.00 s | 7.87 s |
| Peak GPU allocation | 11.82 GiB | 12.25 GiB | 11.10 GiB |
| Windows per second | 87 | 164 | 520 |
| Target region length | 1280 samples | 1280 samples | 64 samples |
| Groups / kernels per group | 8 / 32 | 8 / 32 | 4 / 4 |

Forward includes target construction; these timings exclude input loading and full-trainer bookkeeping.
The head changes the auxiliary objective and filter bank, so lower runtime does not establish equal accuracy.

| Frozen paper source reference | Micro / global batch | Microbatch time | Update time | Windows/s |
|---|---:|---:|---:|---:|
| Current paper settings | 16 / 256 | 0.29 s | 4.61 s | 55 |
| Earlier large-batch capacity check | 512 / 4096 | Not completed | Out of memory | Not measured |

The frozen paper microbatch is much smaller, which explains its shorter batch time.
Its earlier large-batch check exceeded its memory allowance; active paper jobs keep their original settings.

| Validation | Result |
|---|---:|
| Earlier original-source memory allowance | 64 GiB |
| All direct checks: losses, gradients and parameters | Finite |
| All direct checks: process exit | 0 |
| Fixed GPU target error across chunk sizes | 0.0 |
| Paper source and W&B histories | Unchanged |

Shared GPU load can change these timings. Earlier full-trainer bounded stops emitted a shutdown
diagnostic after valid updates; the direct model checks above exited cleanly.
