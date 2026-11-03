# Current EEG-X model

The current experiments use the hybrid EEG-X model selected from the reference baseline. In-domain and cross-domain runs share the same architecture.

## Architecture

| Component | Configuration |
|---|---|
| Encoder | 8 Transformer layers, embedding dimension 64 |
| Attention | 8 heads; head dimension 8; PyTorch SDPA FlashAttention, with math fallback when unsupported |
| Feed-forward network | 64 → 256 → 64, GELU |
| Tokenizer | Non-overlapping 32-sample patches; sum raw projection 32 → 64 and STFT magnitude projection 17 → 64 |
| Position encoding | Fixed physical-electrode XYZ features and sinusoidal time encoding |
| Normalization | RMSNorm throughout, including Q/K; epsilon 1e-6 |
| Predictor / decoder | 1 cross-attention Predictor layer; 4 Transformer decoder layers, width 64 |
| Dropout / masking | Dropout 0.1; mask 75% of patches during pretraining |
| Classification | Mean patch pooling, RMSNorm, dropout 0.1 and a linear head |
| Parameters | 653,626 for pretraining; 402,192 in the encoder |
| Execution | BF16 autocast, FP32 loss/dictionary computation, TF32 matmul; fused Triton RMSNorm and SoftDiCT; compilation disabled |

Recorded GPU attention backends confirm FlashAttention. Classification uses the encoder and head; reconstruction uses the Predictor and decoder too.

## SoftDiCT loss

The model reconstructs the processed EEG waveform. Apply the same fixed, seeded convolution dictionary to the prediction and target. Softmax across competing filters gives differentiable assignments; summing over time gives soft winner counts. Compare prediction and detached target counts with mean squared error.

| Dictionary setting | Value |
|---|---|
| Kernel lengths / dilations | 9 and 15 / 1, 2, 4, 8, 16 |
| Groups / filters per group | 8 / 32 |
| Temperature / features per channel | 1.0 / 2,560 |
| Auxiliary loss weight | 1.0 |

Filters are zero-mean, unit-L2-normalized and fixed. Channels share filters but compete independently. Gradients flow through the predicted signal into the model.

Total loss is masked waveform MSE plus weighted SoftDiCT feature MSE. Waveform MSE uses masked patches; SoftDiCT uses the full predicted window, including visible positions. Counts have no window-length normalization or feature standardization, and the losses have no adaptive balancing.

## Training

| Stage | Optimizer / peak learning rate | Micro / global batch | Maximum epochs |
|---|---|---|---|
| Pretraining | StableAdamW / 2.4e-4 | 256 / 256 | 100 |
| Fine-tuning | AdamW / 1e-3, layer decay 0.75 | 64 / 64 | 20 |
| Linear probe | AdamW / 1e-3; frozen encoder, train-only feature scaler, CPU linear head | 64 / 64 | 20 |

Pretraining selects minimum validation raw MSE. Fine-tuning and probes select maximum validation balanced accuracy. Embedded probes provide validation and diagnostic test trends; downstream final test evaluates the selected checkpoint once.
