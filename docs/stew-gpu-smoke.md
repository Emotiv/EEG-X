# STEW GPU smoke test

Pretraining, embedded linear probing and fine-tuning completed on the full current STEW dataset. This is an engineering check, not a paper result.

| Setting | Value |
|---|---|
| Previous W&B runs removed | 71 |
| New completed runs | 2; engineering intent; test-prefixed names |
| Dataset | 56,928 windows; 48 subjects |
| Saved subject split | Seed 241; 39,138 train, 5,930 validation, 11,860 test windows |
| Model | Depth 8; width 64; 8 heads; BF16; FlashAttention |
| Horizon | 2 pretraining epochs; probes each epoch and on the selected encoder; 2 fine-tuning epochs |
| Batches | Pretraining logical 256 / micro 16; fine-tuning logical 64 / micro 16 |
| Probe classifier | At most 20 epochs; validation balanced-accuracy selection; patience 5 |
| GPU | Physical 0, shared; original model and batch settings retained |

The existing saved subject assignments are reused across both stages. The source encoder is selected on validation reconstruction loss.

| Stage | Selected epoch | Wall minutes | Peak allocated GiB |
|---|---:|---:|---:|
| pretrain | 2 | 16.40 | 0.786 |
| finetune | 2 | 4.91 | 0.124 |

Memory figures measure this process's PyTorch allocations. Shared-device timings do not establish a code speedup.

| Evaluation | Accuracy | Balanced accuracy | Macro F1 | AUROC | Loss |
|---|---:|---:|---:|---:|---:|
| Selected-encoder probe, diagnostic test | 0.6621 | 0.6621 | 0.6617 | 0.7072 | 0.6357 |
| Selected fine-tuned model, final test | 0.6746 | 0.6746 | 0.6744 | 0.7246 | 0.6071 |

Probe test curves remain diagnostic. Fine-tuning selects on validation balanced accuracy and evaluates its selected checkpoint on test once.

| Verification | Result |
|---|---|
| Local and remote metrics | Match saved predictions; finite losses and gradients |
| Checkpoints | Validation selection verified; uploaded best artifacts match local hashes |
| W&B panels | Train/val/model, embedded probe curves, summary-only final test bars and confusion-matrix images |
| Native monitoring | System metrics and nonempty console logs verified on both runs |
| Grouping | No preset groups; stage views separate the runs |
| CPU regression checks | 180 passed; 3 CUDA-only checks skipped; live GPU workflow verified separately |

