# Fresh paper study launch

| Execution | Verified state |
|---|---|
| Reference project | `eegx-bs256`, all 23 old run identities preserved |
| New project | Private `eegx`, fresh run identities |
| Study queue | 81 unique tasks |
| Concurrent execution | 7 tasks per GPU, 14 active across both GPUs |
| Pretraining batch | Microbatch 512, global batch 4096 |
| Pretraining objective | Full-dictionary SoftDiCT with fused execution |
| Dataset assignment | Existing HDF5 releases and 41 saved subject split banks reused |
| Time policy | No fixed stop cap; separate usage meters preserve historical accounting |

Old source snapshots, checkpoints and outputs remain available for reference. The new study uses fresh outputs and selected checkpoints from its own pretraining runs.

| Verification before launch | Result |
|---|---|
| Standards and Spec reviews | 0 open findings after ownership and provenance fixes |
| Final CPU suite | 265 passed, 6 skipped |
| GPU capacity | 7 simultaneous TUAB processes passed on each GPU |
| Integrated GPU smoke | 1 full STEW pretraining epoch, embedded and selected probes, 1 fine-tuning epoch passed |
| Generated queue | All 81 configs and checkpoint dependencies passed preflight |
| Startup audit | All 14 active runs logged finite losses; no tracebacks |

Both review axes cleared the generated queue before launch. Capacity checks establish memory fit, rather than a proportional throughput gain. Final scientific scores still require the complete study.
