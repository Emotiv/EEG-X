# Stage-appropriate run charts

A shared manual classifier layout forced final-test bars onto pretraining pages. Those runs correctly had no final classifier scores, so the bars were empty. The shared overview now discovers charts from actual run history. Scoped classifier views retain their relevant score panels.

| Individual run page | Final classifier test section | Probe trends |
|---|---|---|
| STEW pretraining | Absent | All 11 validation and 11 diagnostic-test trends |
| TUAB source pretraining | Absent | Not part of the shared-source protocol |
| STEW fine-tuning | 11 final scores and confusion matrix | Not an embedded-pretraining run |
| Crowdsourced linear probe | 11 final scores and confusion matrix | Classifier validation history retained |
| Crowdsourced fine-tuning | 11 final scores and confusion matrix | Not an embedded-pretraining run |

The original empty-panel symptom was reproduced and then removed on the same pretraining URL. Fresh run pages were also checked in an authenticated browser.

| Verification | Result |
|---|---:|
| Fresh engineering runs | 5 finished |
| Individual browser checks | 5 passed |
| Focused regression checks | 26 passed |
| Reference runs retained | 23 |

The remaining tests ran on GPU one with explicit sharing after an unrelated job occupied GPU zero. Completed pretraining was retained. The unrelated job was left running, and the full paper queue remains stopped for user review.
