"""Predict fixed channel-wise HYDRA patch features without transform backward."""

import torch
from torch import nn
import torch.nn.functional as F

from pretrain.models.hydra import ChannelWiseHydraMultivariateGPU


class HydraPatchTarget(nn.Module):
    """Auxiliary decoder head with detached, locally computed HYDRA targets.

    Unlike a feature loss on reconstructed EEG, the prediction already lives in
    feature space. Hard winner counts therefore require no surrogate derivative.
    Each target region uses only its own channel and non-overlapping samples.
    """

    def __init__(self, width, objective, *, max_response_elements=32 * 1024 * 1024):
        super().__init__()
        if type(max_response_elements) is not int or max_response_elements < 1:
            raise ValueError("HYDRA target response limit must be a positive integer")
        self.max_response_elements = max_response_elements
        self.patch_size = objective["patch_size"]
        self.transform = ChannelWiseHydraMultivariateGPU(
            input_length=self.patch_size,
            num_channels=1,
            k=objective["k"],
            g=objective["groups"],
            kernel_size=objective["kernel_size"],
            seed=objective["seed"],
        )
        self.feature_dim = self.transform.num_features
        self.head = nn.Linear(width, self.feature_dim)

    @torch.no_grad()
    def targets(self, eeg):
        batch, channels, samples = eeg.shape
        regions = samples // self.patch_size
        windows = eeg.float().reshape(batch, channels, regions, self.patch_size)
        windows = windows.permute(0, 2, 1, 3).reshape(-1, 1, self.patch_size)
        # Bound temporary convolution responses without caching signals or targets.
        response_size = self.patch_size * self.transform.h * self.transform.k
        chunk = max(1, self.max_response_elements // response_size)
        with torch.autocast(device_type=eeg.device.type, enabled=False):
            features = torch.cat([self.transform(part) for part in windows.split(chunk)])
        return features.reshape(batch, regions, channels, self.feature_dim)

    def forward(self, eeg, decoded, mask):
        batch, channels, samples = eeg.shape
        regions, group = samples // self.patch_size, self.patch_size // 32
        tokens = decoded.reshape(batch, regions, group, channels, decoded.shape[-1]).mean(2)
        prediction = self.head(tokens)
        region_mask = mask.reshape(batch, regions, group, channels).amax(2)
        errors = F.mse_loss(prediction.float(), self.targets(eeg), reduction="none").mean(-1)
        return (errors * region_mask).sum() / region_mask.sum().clamp_min(1)
