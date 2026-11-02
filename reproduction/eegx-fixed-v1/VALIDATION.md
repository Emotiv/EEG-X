# Fixed-split release validation

| Check | Result |
|---|---:|
| Automated tests passed | 293 |
| Automated tests skipped | 1 |
| Audited dataset files | 8 |
| Audited EEG windows | 772,784 |
| Self-contained HDF5 payload, bytes | 42,173,621,909 |
| Measured GPU workload cases | 17 |
| Minimum per-task reservation, MiB | 1,338 |
| Maximum per-task reservation, MiB | 6,874 |
| Remaining Standards findings | 0 |
| Remaining Spec findings | 0 |

All EEG windows passed their original signal-hash checks. The new release preserves signal values and sample multiplicity while changing split policy.

| Protocol check | Result |
|---|---|
| Official TUAB/TUEV test membership | Preserved |
| TUAB source test subjects | Excluded |
| Fixed split with different training seeds | Verified through actual pretraining and fine-tuning |
| Historical STEW/DREAMER split identities | Matched to reference run artifacts |
| Full selection evidence | Included and hashed in upload bundle |
| Memory receipts | Bound to resolved workload, data identity, dimensions and GPU hardware |
| Source lint | Passed |
| Existing notebook lint | Unrelated errors remain in the original position-plotting notebook |

STEW and DREAMER split selection used historical test scores. Their results remain exploratory; new checkpoint selection uses validation only. GPU measurements cover updates and embedded encoder feature extraction, not completed experiment scores.
