# Supplied EEG-X manuscript check

Checked the author-supplied manuscript locally, including the rendered architecture
and scaling table. The manuscript is not included in this code release. This supersedes the earlier
comparison with the public arXiv version. The user subsequently selected the proposal
as the local default; the last column records that selection.

| Point | Supplied manuscript | Selected local default |
|---|---|---|
| Architecture and scaling | Appendix C.1 and Table 8, p. 18: encoder depth 8, decoder depth 4, heads 8; table widths 64/128/512 and depths 4/8. Prose also says width 16, inconsistent with the table. | Train width 64/depth 8 only. The automatic model-size sweep has been removed. |
| Normalization | RMSNorm, LayerNorm, and Q/K normalization are not specified. | All-RMSNorm and Q/K normalization are local model choices. |
| Masking | MAE visible-token encoding is described. SSP, a specific sampling algorithm, and a 75% mask ratio are not specified. | Scattered deterministic masks are a local choice; the earlier claim that this manuscript requires SSP is withdrawn. |
| DiCT definition | Section 2.3: argmax competition and temporal winner counts. Appendix B.3, p. 16: default G=8, K=32; kernel lengths 9/15; dilations 1/2/4/8/16. Soft routing is not specified. | Keep that dictionary and temporal counts with softmax assignments, as approved. Retain additive raw MSE as the selected local objective; it is not specified by the manuscript. |
| Spatial encoding | Equation 3, p. 4: separable sinusoidal encoding with frequency base 1000. | Exact EEG-X2 fixed normalized XYZ features, base 5; explicit local difference. |
| Reconstruction targets | Section 2.2 and Appendix B.5 specify rASR-cleaned targets. Appendix C.4 still describes ICA processing. | No rASR in input or target; reconstruct the same processed signal. User-selected difference; source processing still needs evidence. |
| Training recipe | Appendix C.2, p. 18: Adam, learning rate 1e-3, cosine schedule, batch 256, pretraining 300 epochs, fine-tuning 10 epochs. Separate fine-tuning optimizer details are not specified. | Keep selected pretraining 100 epochs and fine-tuning 20 epochs, with StableAdamW/AdamW and warmup/cosine. |
| Subject splits | Appendix C.4, p. 19: official TUH splits; other datasets use 20% subject-wise test data. Exact seeds and validation proportion are not specified. | Navid explicitly requested pooled subject splits for 5 seeds. The local ratio 70/10/20 remains an implementation choice. |
| Teacher and latent losses | EMA teacher, latent alignment, and covariance/variance regularization objectives are not specified for EEG-X. | Their omission is not the previously claimed mismatch with this manuscript. No need to add them based on the older paper. |
| Tokenizer | Section 2.1 projects STFT magnitudes. Appendix C.1, p. 18 specifies patch length 128 and overlap 32. An added raw projection is not specified. | Current raw-plus-STFT fusion and non-overlapping patches of length 32 differ. |

The new manuscript resolves the missing-table question and removes the old SSP
and teacher-loss objections. The selected primary model follows the table. Target, tokenizer,
position, and objective differences are deliberate user-selected defaults.

Unspecified details use recorded defaults under Navid's instruction. Preserve the
draft inconsistencies and selected deviations in reporting. Verify source evidence
before paper runs and do not reuse historical scores for revised configurations.
