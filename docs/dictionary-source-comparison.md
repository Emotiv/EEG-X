# Original dictionary source comparison

The original dictionary modules run on the same current backbone and optimizer, using identical TUAB signals and masks. Main DiCT keeps its original constructor defaults and dictionary math.

| Implementation | Micro/global 512/4096, update seconds | Micro/global 16/4096, update seconds |
|---|---:|---:|
| Original main DiCT, STE | Tensor-index limit | 51.915 |
| Original SoftDiCT, native convolution | Out of memory at 48 GiB | 29.255 |
| Bounded SoftDiCT Torch | 14.829 | 22.371 |
| Optimized SoftDiCT | 3.102 | 14.148 |
| HYDRA dense | 2.427 | 14.350 |
| Historical bounded Torch, shared GPU load | 46.86 | Not measured |

Both original sources fail at the larger microbatch in this configuration. Earlier shared-load and fresh idle-start measurements reflect different GPU conditions.

| Source at micro/global 16/4096 | Forward / micro, ms | Backward / micro, ms | Optimizer / update, ms | Peak process allocation, GiB |
|---|---:|---:|---:|---:|
| Original main DiCT, STE | 134.30 | 67.97 | 7.67 | 11.20 |
| Original SoftDiCT, native convolution | 61.68 | 52.09 | 7.45 | 5.26 |
| Bounded SoftDiCT Torch | 42.64 | 44.44 | 7.69 | 1.94 |
| Optimized SoftDiCT | 23.51 | 31.51 | 7.40 | 0.83 |
| HYDRA dense | 26.47 | 29.58 | 7.16 | 0.84 |

Main DiCT spends most time in forward; optimizer time is small. Main DiCT and HYDRA dense change the feature objective.

The HYDRA small-microbatch measurement is a separate follow-up. Its small timing gap from optimized SoftDiCT is within observed run variation.
