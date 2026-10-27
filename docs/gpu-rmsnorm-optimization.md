# Fused GPU RMSNorm

The optional backend keeps the original RMSNorm equation and first-order training gradients. The first benchmark selected it for TUAB. The current paper configs select it for all datasets at the user's request.

| Real-data workload | Native RMSNorm | Fused RMSNorm | Change |
|---|---:|---:|---:|
| TUAB seed 241, model | 2.821 s | 2.418 s | 14.3% faster |
| TUAB seed 242, model | 2.845 s | 2.439 s | 14.3% faster |
| TUAB seed 243, model | 2.840 s | 2.430 s | 14.4% faster |
| TUEV, model | 0.605 s | 0.560 s | 7.3% faster |
| STEW first check, model | 0.496 s | 0.502 s | 1.2% slower |
| STEW repeat, model | 0.500 s | 0.542 s | 8.5% slower |
| TUAB integrated, model | 2.849 s | 2.450 s | 14.0% faster |
| TUAB training loop with loader | 3.109 s | 2.634 s | 15.3% faster |

Model timings are interleaved; the loader comparison is sequential and short. The STEW measurements did not show a speedup. The current configs still enable it at the user's request.

| GPU profile per update | Before both changes | Fused RMSNorm and allocation option |
|---|---:|---:|
| RMSNorm kernel time | 433.1 ms | 44.7 ms |
| GPU kernels | 29,110 | 16,563 |
| Peak training allocation | 10.82 GiB | 10.21 GiB |

The kernel-count comparison includes the earlier allocation-fill option. These are short-profile measurements, not a full-study runtime.

| Validation | Result |
|---|---:|
| RMS calculation precision | FP32 |
| Final CUDA operator cases / width checks | 57 / 8 passed |
| Final GPU dispatch and fallback cases | 16 passed |
| Real full-microbatch model cases / TUAB seeds | 7 / 5 |
| Maximum model-gradient relative difference | 1.418e-06 |
| Accumulated-update parameter maximum difference | 1.164e-10 |
| Final model fused calls / gradient tensors | 53 / 216 |
| Supported RMSNorm feature widths | 8, 12, 16, 32, 64, 96, 128 |
| Width-one gradient check | Native fallback after tolerance failure |
| Unsupported widths, layouts and CPU | Native fallback |
| Higher-order gradients on fused CUDA path | Not supported |
| CPU regression suite before inference checks | 250 passed, 4 CUDA-only skipped |
| Added CUDA inference checks | 2 not yet run |
| CPU checkpoint transfer | Linear probe and fine-tune passed |

The final dispatch guard was added after the full-update timings; it retains the measured model widths and leaves GPU arithmetic unchanged. Final operator and model checks passed on an identical GPU model.

Validation, epoch probes, online tracking and GPU checkpoint resume were not measured. The forced profiling stop emitted a native cleanup diagnostic, while the parent exited successfully and saved a finite checkpoint.
