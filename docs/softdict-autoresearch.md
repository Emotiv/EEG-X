# SoftDiCT autoresearch snapshot

The accepted code preserves the agreed differentiable SoftDiCT objective and full
paper dictionary. Active paper training, saved splits and the aggregate cap are unchanged.

| Verified model-update workload | Reference | Accepted code | Observed reduction |
|---|---:|---:|---:|
| TUAB fused reference, micro / global 512 / 4096 | 25.13 s | 9.87 s | 60.7% |
| STEW fused reference, 512 / 4096 | 3.93 s | 1.76 s | 55.1% |
| TUEV fused reference, 512 / 4096 | 5.19 s | 2.19 s | 57.8% |
| TUAB original implementation, 16 / 256 | 5.32 s | 1.64 s | 69.1% |

Interleaved shared-GPU timings exclude loading. These are engineering gains,
not isolated hardware speedup, downstream accuracy or literal hard-count manuscript equivalence.

| Fidelity and regression evidence | Result |
|---|---:|
| Dictionary seeds and exact buffer roundtrips verified | 241, 242, 243, 244, 245 |
| General-case maximum feature / input-gradient difference | 4.58e-5 / 3.26e-9 |
| Full BF16 model loss / relative parameter-gradient difference | 2.86e-6 / 2.10e-6 |
| Specialized fold versus native operation | 120 cases; bitwise identical |
| Full CPU suite on immutable accepted source | 235 passed; 4 GPU-only skipped |
| Independent Standards / Spec findings | 0 / 0 |
| Peak accepted PyTorch allocation, TUAB 512 / 4096 | 10.83 GiB |

Numerical tolerances, precision and RNG policies remained fixed. Real HDF5 training
and finite-checkpoint checks passed; the accepted patch has not been deployed or published.
