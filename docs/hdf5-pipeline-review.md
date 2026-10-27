# Regenerated dataset review

All datasets are regenerated and audited. EEG-X2 preprocessing preserves original overlap and sample multiplicity; duplicate-linked subjects share a partition.

| Dataset | Old samples | New samples | New class counts |
|---|---:|---:|---|
| DREAMER arousal | 170,246 | 170,246 | 88,611 / 81,635 |
| STEW | 56,928 | 56,928 | 28,464 / 28,464 |
| TUAB | 409,455 | 409,455 | 203,616 / 205,839 |
| TUEV binary | 98,684 | 98,684 | 73,372 / 25,312 |
| Crowdsourced | 13,363 | 12,305 | 6,153 / 6,152 |
| BCICIV-2a | 5,184 | 5,184 | 1,296 / 1,296 / 1,296 / 1,296 |
| Kalunga / SSVEP | 960 | 960 | 240 / 240 / 240 / 240 |
| PhysioNetMI | 19,022 | 19,022 | 2,455 / 2,480 / 2,438 / 2,465 / 9,184 |

Counts pool all partitions and follow stored numeric class order. DREAMER uses the agreed binary conversion of its original ratings.

| Recorded correction | Result |
|---|---|
| Crowdsourced | 13,363 → 12,305: 963 fewer windows from rate correction and 95 from label boundaries |
| PhysioNet duration | 582 reference windows at 128 Hz: historical 2.5 s becomes true 2 s; sample and class totals unchanged |
| Kalunga labels | Corrected legacy 17/21 Hz frequency meanings; all 4 classes retain 240 samples |

These are explicit corrections. No windows were inserted to force a reference count; legacy waveform replay establishes sample identity, while new signals use EEG-X2.

| Verification | Result |
|---|---|
| Standards / integrity | 1 finding fixed; 0 remaining |
| Requirements | 5 findings fixed; 0 remaining |
| CPU tests and static checks | 178 passed; 3 CUDA checks skipped; static checks passed |
| Signal integrity | Every stored window hash and recorded raw-source digest verified; complete source-record replay matches per dataset |
| Split integrity | 5 banks per dataset, seeds 241–245; class, subject and duplicate-group checks pass; separate grouped TUAB 80/20 source bank |
| Established original-row lineage | Embedded artifact hashes, original array lengths and field/row candidates; complete row coverage and native crop bounds verified |
| Legacy waveform identity | All 5,184 BCICIV, 960 Kalunga and 19,022 PhysioNet reference windows replay successfully |
| DataLoader and model | Fully drained worker checks match; raw and Soft DiCT CPU gradients pass for every dataset |

Original source aliases remain explicit. These checks establish data and pipeline correctness under the selected protocol; paper scores require training.
