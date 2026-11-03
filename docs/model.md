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

The larger comparison keeps the same architecture and training recipe while increasing embedding width. Parameter counts below are measured from this implementation and do not match the paper's reported totals.

| W&B size | Encoder layers | Embedding width | Encoder parameters | Total pretraining parameters |
|---|---:|---:|---:|---:|
| `d8w64` | 8 | 64 | 402,192 | 653,626 |
| `d8w512` | 8 | 512 | 25,237,520 | 41,011,258 |

Group runs by `domain`, `stage`, `task_name` and `size` to average training seeds separately for each model size. Runs keep an unset W&B group string; `size` is a configuration field.

## SoftDiCT loss

The model reconstructs the processed EEG waveform. Apply the same fixed, seeded convolution dictionary to the prediction and target. Softmax across competing filters gives differentiable assignments; summing over time gives soft winner counts. Compare prediction and detached target counts with mean squared error.

| Dictionary setting | Value |
|---|---|
| Kernel lengths / dilations | 9 and 15 / 1, 2, 4, 8, 16 |
| Groups / filters per group | 8 / 32 |
| Temperature / features per channel | 1.0 / 2,560 |
| Auxiliary loss weight | 1.0 |
| Dictionary random seed | 245, shared across training seeds |

Filters are zero-mean, unit-L2-normalized and fixed. Channels share filters but compete independently. Gradients flow through the predicted signal into the model.

Total loss is masked waveform MSE plus weighted SoftDiCT feature MSE. Waveform MSE uses masked patches; SoftDiCT uses the full predicted window, including visible positions. Counts have no window-length normalization or feature standardization, and the losses have no adaptive balancing.

### How the differentiable dictionary works

For each EEG channel, convolve the waveform with each fixed filter bank using symmetric zero padding and the configured dilation. Within each group, filters compete at every time sample. Replace a discrete winner with a softmax probability:

```text
response[t, g, k] = convolution(signal, filter[g, k])[t]
probability[t, g, k] = softmax_k(response[t, g, k] / temperature)
count[g, k] = sum_t probability[t, g, k]
features(signal) = concatenate counts across kernel lengths, dilations and channels
```

Each count measures how often a filter responds strongly relative to its group. The current objective uses positive-response probabilities only. It has no separate negative-winner branch, amplitude weighting or first-difference branch. Pooling removes temporal order from the dictionary features; waveform MSE still checks sample alignment.

Apply this transform to both the decoder waveform and the original processed EEG:

```text
target_features = stop_gradient(features(original_eeg))
dict_loss = mean((features(reconstructed_eeg) - target_features)^2)
total_loss = masked_waveform_mse + auxiliary_weight * dict_loss
```

The feature MSE averages over the batch and all channel/filter features. Soft assignments let gradients pass through temporal pooling, softmax and convolution into the reconstructed waveform, decoder, Predictor and encoder. The filters remain fixed buffers and receive no updates. The target branch runs without gradients.

### What `soft_dict_mse` means, compared with hard DiCT

`soft_dict_mse` is the squared difference between two vectors of soft counts. It compares dictionary features of reconstructed EEG against dictionary features of the original EEG. It does not compare filter weights or softmax probabilities directly at each time sample.

| Operation | Hard winner-count DiCT | Current SoftDiCT |
|---|---|---|
| Assignment within a filter group | Only the largest response wins | Softmax distributes probability across filters |
| Temporal pooling | Sum discrete winner indicators | Sum soft assignment probabilities |
| Reconstruction loss | MSE between hard-count vectors | MSE between soft-count vectors |
| Gradient to the waveform | Discrete argmax/count routing blocks a useful gradient | Softmax and convolution propagate gradients |
| Relationship | Hard dictionary statistic | Differentiable surrogate, not numerically identical to hard DiCT |

SoftDiCT preserves the fixed convolution dictionary and temporal counting idea. Softening changes the statistic and therefore the loss. This comparison concerns hard winner counts; the current implementation does not include the extra branches of other HYDRA variants.

```python
def hard_dict(signal):
    # Same filters, channel handling and pooling as soft_dict below.
    features = []
    for filters, dilation in fixed_filter_banks:
        responses = same_padded_convolution(signal.float(), filters, dilation)
        responses = group_filters(responses)  # [batch, channel, time, group, filter]
        winners = responses.argmax(dim="filter")
        assignments = one_hot(winners, num_classes=filters_per_group).float()
        hard_counts = assignments.sum(dim="time")
        features.append(flatten_channel_group_filter(hard_counts))
    return concatenate(features, dim="feature")


def soft_dict(signal):
    # Process channels independently with the same fixed filter banks.
    features = []
    for filters, dilation in fixed_filter_banks:
        responses = same_padded_convolution(signal.float(), filters, dilation)
        responses = group_filters(responses)  # [batch, channel, time, group, filter]
        probabilities = softmax(responses / temperature, dim="filter")
        soft_counts = probabilities.sum(dim="time")
        features.append(flatten_channel_group_filter(soft_counts))
    return concatenate(features, dim="feature")


with torch.no_grad():
    target_counts = soft_dict(original_eeg)
predicted_counts = soft_dict(reconstructed_eeg)
soft_dict_mse = (predicted_counts - target_counts).square().mean()

# Comparison only: argmax interrupts gradients to reconstructed_eeg.
with torch.no_grad():
    hard_target_counts = hard_dict(original_eeg)
hard_predicted_counts = hard_dict(reconstructed_eeg)
hard_dict_mse = (hard_predicted_counts - hard_target_counts).square().mean()
```

The named dimensions above are pseudocode. `hard_dict` isolates the hard winner-count rule for comparison; training uses `soft_dict`. Both functions share convolution and pooling. Their assignment step differs: `argmax` followed by `one_hot` versus `softmax`.

The final mean covers every sample, channel, filter group, kernel length and dilation. Counts are summed over time before squaring, so longer windows can change the loss scale even with the same auxiliary weight.

The fused GPU implementation combines convolution, softmax and temporal reduction to reduce temporary memory. It retains this objective and computes dictionary features and loss in FP32. SoftDiCT therefore acts as an auxiliary waveform reconstruction loss; it is not the model's input tokenizer in this configuration.

### High-level model code

This pseudocode shows the current pretraining path. Selection and restoration helpers preserve each sample's original patch order. `mask` is true for hidden patches; the current auxiliary weight is fixed as shown below.

```python
auxiliary_weight = 1.0  # Fixed current configuration, not learned.

def forward(eeg, channel_ids, mask, auxiliary_weight):
    # Tokenize each patch independently, then add spatial/time positions.
    patches = patchify(eeg)
    spectrum = abs(rfft(patches.float() * hann_window, norm="ortho"))
    positions = spatial_time_positions(channel_ids)
    tokens = rmsnorm(raw_projection(patches)
                     + spectral_projection(spectrum) + positions)

    # Encode visible patches and predict representations at hidden positions.
    context = encoder(select_visible(tokens, mask))
    queries = mask_token + select_hidden(positions, mask)
    hidden = predictor(context, queries)
    full_sequence = restore_patch_order(context, hidden, mask)
    decoded = decoder(full_sequence + positions)
    predicted_patches = reconstruction_head(decoded)

    # Waveform loss covers hidden patches only.
    patch_mse = (predicted_patches.float() - patches.float()).square().mean(-1)
    raw_loss = (patch_mse * mask).sum() / mask.sum()

    # Dictionary loss covers the full reconstructed waveform, per channel.
    reconstructed_eeg = unpatchify(predicted_patches)
    with torch.no_grad():
        target_features = soft_dict(eeg.float())
    predicted_features = soft_dict(reconstructed_eeg.float())
    dict_loss = mse_loss(predicted_features, target_features)
    loss = raw_loss + auxiliary_weight * dict_loss
    return loss, raw_loss.detach(), dict_loss.detach()


optimizer.zero_grad()
loss, raw_mse, soft_dict_mse = model(eeg, channel_ids, mask, auxiliary_weight)
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
optimizer.step()
```

Both loss terms use the fixed unit weight shown in the dictionary settings table. Thus, `loss = raw_loss + dict_loss`. The auxiliary weight multiplies the dictionary MSE directly; there is no learned weighting or automatic scale correction. Validation logs both terms, but checkpoint selection uses raw waveform MSE.

## Training

| Stage | Optimizer / peak learning rate | Micro / global batch | Maximum epochs |
|---|---|---|---|
| Pretraining | StableAdamW / 2.4e-4 | 256 / 256 | 100 |
| Fine-tuning | AdamW / 1e-3, layer decay 0.75 | 64 / 64 | 20 |
| Linear probe | AdamW / 1e-3; frozen encoder, train-only feature scaler, CPU linear head | 64 / 64 | 20 |

Pretraining selects minimum validation raw MSE. Fine-tuning and probes select maximum validation balanced accuracy. Embedded probes provide validation and diagnostic test trends; downstream final test evaluates the selected checkpoint once.
