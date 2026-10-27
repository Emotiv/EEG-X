# Optimized SoftDiCT equivalence audit

Optimized SoftDiCT preserves the original mathematical objective for standard training. Its execution and supported backend API differ.

| Property | Original native | Optimized Triton |
|---|---|---|
| Groups / kernels per group | 8 / 32 | Same |
| Kernel lengths | 9, 15 | Same |
| Dilations | 1, 2, 4, 8, 16 | Same |
| Temperature / feature-loss weight | 1 / 1 | Same |
| Fixed seeded centered unit-L2 filters | Saved buffers | Bitwise identical |
| Features | Group softmax, temporal sums | Same |
| Loss | Masked raw MSE plus feature MSE; detached targets | Same |
| Dictionary precision | FP32 | FP32 |
| Autograd support | Standard native autograd | First-order only |
| Dictionary sizes | Any validated geometry | Power-of-two groups/kernels; product at most 256 |

The selected dictionary fits both backends. GPU execution changes include fusion, bounded buffers, recomputation and reduction order.

| Fresh direct comparison with original source | Result |
|---|---:|
| Cases / seeds | 13 / 5 |
| Maximum feature absolute difference | 1.526e-05 |
| Maximum input-gradient absolute difference | 7.276e-10 |
| Maximum feature-MSE absolute difference | 1.788e-07 |
| Full BF16 model loss absolute difference | 2.861e-06 |
| Full model parameter-gradient relative difference | 2.091e-06 |
| Compared model gradient tensors | 216 |
| Raw reconstruction loss / RNG state | Exactly equal |

All checks pass at unchanged tolerances. This proves tested first-order numerical parity; it does not establish bitwise identity or identical final experiment scores.
