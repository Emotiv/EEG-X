"""EEG-X hybrid encoder and masked reconstruction with the active Predictor."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pretrain.models.attention import CrossAttnTRMBlock, MultiHeadAttention, TransformerBlock


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        d_model = config["emb_size"]
        attn_heads = config["num_heads"]
        d_ffn = config["dim_ff"]
        layers = config["layers"]
        dropout = config["dropout"]
        enable_res_parameter = True
        # TRMs
        self.TRMs = nn.ModuleList(
            [
                TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout)
                for i in range(layers)
            ]
        )

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


class Predictor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter)
                for i in range(layers)
            ]
        )

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


class EEGXEncoder(nn.Module):
    """Hybrid patch tokenizer and the existing EEG-X context encoder.

    Tokens use time-major, channel-minor order. Channel identity is explicit,
    and the tokenizer accepts different task durations without changing weights.
    """

    def __init__(self, width=64, depth=8, heads=8, dropout=0.1):
        super().__init__()
        from pretrain.channel_identity import load_channel_vocabulary
        from pretrain.normalization import RMSNorm
        from pretrain.models.position import FixedChannelEmbedding

        self.embed_dim = width
        self.raw_projection = nn.Linear(32, width)
        self.spectral_projection = nn.Linear(17, width)
        self.register_buffer("stft_window", torch.hann_window(32))
        vocabulary = load_channel_vocabulary()
        self.channel_embedding = FixedChannelEmbedding(vocabulary, width)
        self.norm = RMSNorm(width)
        self.contex_encoder = Encoder(
            {
                "emb_size": width,
                "num_heads": heads,
                "dim_ff": width * 4,
                "layers": depth,
                "dropout": dropout,
            }
        )
        _prepare_paper_transformer(self.contex_encoder)

    @staticmethod
    def patchify(eeg):
        batch, channels, samples = eeg.shape
        return eeg.reshape(batch, channels, samples // 32, 32).permute(0, 2, 1, 3).flatten(1, 2)

    def positions(self, channel_ids, patches):
        batch, channels = channel_ids.shape
        time = torch.arange(patches, device=channel_ids.device, dtype=torch.float32)
        frequency = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=channel_ids.device).float()
            * (-np.log(10000.0) / self.embed_dim)
        )
        phases = time[:, None] * frequency[None, :]
        temporal = torch.stack((phases.sin(), phases.cos()), -1).flatten(-2)
        spatial = self.channel_embedding(channel_ids)
        return (spatial[:, None] + temporal[None, :, None]).reshape(
            batch, patches * channels, self.embed_dim
        )

    def embed_patches(self, eeg, channel_ids):
        return self.tokenize(eeg, channel_ids)[0]

    def tokenize(self, eeg, channel_ids):
        """Return patch embeddings and their shared position encoding."""
        if eeg.ndim != 3 or eeg.shape[-1] < 32 or eeg.shape[-1] % 32:
            raise ValueError("EEG requires complete non-overlapping patches of 32 samples")
        if channel_ids.ndim == 1:
            channel_ids = channel_ids.expand(eeg.shape[0], -1)
        if channel_ids.shape != eeg.shape[:2]:
            raise ValueError("Channel identities must match the EEG batch")
        positions = self.positions(channel_ids, eeg.shape[-1] // 32)
        patches = self.patchify(eeg)
        # One FFT per patch is the specified center=False STFT with hop 32.
        # Keeping the FFT inside each patch prevents hidden-sample leakage.
        spectrum = torch.fft.rfft(patches.float() * self.stft_window, norm="ortho").abs()
        tokens = self.raw_projection(patches) + self.spectral_projection(spectrum)
        return self.norm(tokens + positions), positions

    def forward_features(self, eeg, channel_ids):
        return self.contex_encoder(self.embed_patches(eeg, channel_ids)).mean(1)

    def forward_emotiv_features(self, eeg, channel_ids, *, input_window_normalization="none"):
        """Keep the restored training interface without adding signal normalization."""
        if input_window_normalization != "none":
            raise ValueError("Model-side window normalization is outside the EEG-X contract")
        return self.forward_features(eeg, channel_ids)


def _prepare_paper_transformer(module):
    """Reuse EEG-X blocks with fused RMSNorm and explicit SDPA dispatch."""
    from pretrain.normalization import RMSNorm

    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, RMSNorm(child.normalized_shape[-1]))
        else:
            _prepare_paper_transformer(child)
    if isinstance(module, MultiHeadAttention):
        module.use_sdpa = True
        module.qk_normalization = "rmsnorm"


class EEGXReconstruction(nn.Module):
    """Revised EEG-X reconstruction using its active cross-attention Predictor.

    The old teacher's outputs did not enter the reconstruction objective. This
    path omits that unused computation, keeps Predictor gradients, and adds the
    specified Transformer decoder before the raw patch projection.
    """

    def __init__(self, config):
        super().__init__()
        import copy
        from pretrain.channel_identity import load_channel_vocabulary
        from pretrain.models.dictionary import SoftDiCT
        from pretrain.objectives import validate_objective

        width, depth, heads = config["width"], config["depth"], config.get("heads", 8)
        self.input_size = config["input_samples"]
        if self.input_size < 32 or self.input_size % 32:
            raise ValueError("Input duration must contain complete patches")
        dropout = config.get("dropout", 0.1)
        self.encoder = EEGXEncoder(width, depth, heads, dropout)
        self.mask_token = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.Predictor = Predictor(width, heads, width * 4, True, 1)
        self.Decoder = Encoder(
            {
                "emb_size": width,
                "num_heads": 8,
                "dim_ff": width * 4,
                "layers": 4,
                "dropout": dropout,
            }
        )
        _prepare_paper_transformer(self.Predictor)
        _prepare_paper_transformer(self.Decoder)
        self.reconstruction_head = nn.Linear(width, 32)
        objective = config.get("objective")
        validate_objective(objective)
        self.hydra_weight = objective["weight"] if objective else 0.0
        self.hydra = None if objective is None else SoftDiCT(objective)
        self.contract = {
            "name": "eegx_hybrid_reconstruction_v2",
            "width": width,
            "depth": depth,
            "heads": heads,
            "decoder_depth": 4,
            "predictor_depth": 1,
            "teacher": "none",
            "dropout": dropout,
            "input_samples": self.input_size,
            "patch_size": 32,
            "patch_stride": 32,
            "token_order": "time_channel",
            "fusion": "raw_plus_stft_magnitude",
            "stft": {
                "n_fft": 32,
                "hop": 32,
                "window": "hann_periodic",
                "center": False,
                "normalized": True,
            },
            "position": {
                "spatial": copy.deepcopy(self.encoder.channel_embedding.encoding_contract),
                "temporal": "sinusoidal_base_10000_v1",
            },
            "norm": "all_rms_fp32_eps_1e-6",
            "attention": "flash_else_sdpa_math",
            "objective": copy.deepcopy(objective),
            "raw_loss": "masked_patch_mse",
            "hydra_loss": "full_reconstruction_soft_winner_counts_mse",
            "reconstruction_target": "same_processed_input",
            "window_normalization": "none",
            "mask_ratio": 0.75,
            "mask_version": "paper_identity_mix31_v1",
            "vocabulary_sha256": load_channel_vocabulary().fingerprint,
        }

    def get_extra_state(self):
        import copy

        return copy.deepcopy(self.contract)

    def set_extra_state(self, state):
        if state != self.contract:
            raise ValueError("Incompatible EEG-X checkpoint contract")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if state_dict.get("_extra_state") != self.contract:
            raise ValueError("Incompatible EEG-X checkpoint contract")
        for name, expected in self.named_buffers():
            if name not in state_dict or not torch.equal(expected.cpu(), state_dict[name].cpu()):
                raise ValueError(f"Incompatible EEG-X fixed buffer: {name}")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def parameter_counts(self):
        encoder = sum(p.numel() for p in self.encoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": encoder, "decoder_adapter": total - encoder, "total": total}

    def predict_masked(self, eeg, channel_ids, *, mask, **kwargs):
        if eeg.shape[-1] != self.input_size:
            raise ValueError("Pretraining duration differs from checkpoint contract")
        tokens, positions = self.encoder.tokenize(eeg, channel_ids)
        if mask.shape != tokens.shape[:2] or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("A binary mask must match the token layout")
        visible_count = tokens.shape[1] // 4
        if visible_count < 1 or not ((mask == 0).sum(1) == visible_count).all():
            raise ValueError("Each sample requires exactly 75 percent masked patches")
        order = mask.argsort(dim=1, stable=True)
        visible_ids, hidden_ids = order[:, :visible_count], order[:, visible_count:]
        visible = tokens.gather(1, visible_ids[..., None].expand(-1, -1, self.encoder.embed_dim))
        encoded = self.encoder.contex_encoder(visible)
        masked = self.mask_token + positions.gather(
            1, hidden_ids[..., None].expand(-1, -1, self.encoder.embed_dim)
        )
        predicted = self.Predictor(encoded, masked)
        decoded = torch.cat((encoded, predicted), 1).gather(
            1, order.argsort(1)[..., None].expand(-1, -1, self.encoder.embed_dim)
        )
        reconstruction = self.Decoder(decoded + positions)
        return self.reconstruction_head(reconstruction), mask

    def patch_errors(self, eeg, prediction):
        return (prediction.float() - self.encoder.patchify(eeg).float()).square().mean(-1)

    def forward(self, eeg, channel_ids, *, mask):
        prediction, mask = self.predict_masked(eeg, channel_ids, mask=mask)
        raw = (self.patch_errors(eeg, prediction) * mask).sum() / mask.sum()
        hydra = raw.new_zeros(())
        if self.hydra is not None:
            batch, channels, samples = eeg.shape
            reconstruction = prediction.reshape(batch, samples // 32, channels, 32)
            reconstruction = reconstruction.permute(0, 2, 1, 3).reshape(
                batch * channels, 1, samples
            )
            with torch.no_grad():
                target = self.hydra(eeg.float().reshape(batch * channels, 1, samples))
            hydra = F.mse_loss(self.hydra(reconstruction.float()), target)
        return {
            "loss": raw + self.hydra_weight * hydra,
            "mae_loss": raw.detach(),
            "hydra_loss": hydra.detach(),
            "pred": prediction,
            "mask": mask,
        }
