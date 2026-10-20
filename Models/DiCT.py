import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseDiCTMultivariateGPU(nn.Module):
    """
    Channel-wise DiCT transformation using max-response features only.
    Forward pass:
        Uses hard argmax winner assignments.
    Backward pass:
        Uses standard softmax gradients through a straight-through estimator.
    """

    def __init__(
        self,
        input_length: int,
        num_channels: int,
        k: int = 8,
        g: int = 32,
        max_num_channels: int = 14,
        seed: int | None = None,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if input_length < 9:
            raise ValueError("input_length must be at least 9.")

        self.input_length = input_length
        self.num_channels = num_channels
        self.max_num_channels = max_num_channels
        self.k = k
        self.g = g

        kernel_length = 9
        max_exponent = np.log2((input_length - 1) / (kernel_length - 1))
        dilations = 2 ** torch.arange(int(max_exponent) + 1, dtype=torch.long)
        paddings = torch.div((kernel_length - 1) * dilations, 2, rounding_mode="floor").to(torch.long)

        self.register_buffer("dilations", dilations)
        self.register_buffer("paddings", paddings)

        self.num_dilations = len(dilations)

        # Fixed random convolutional kernels.
        W = torch.randn(self.num_dilations, self.k * self.g, kernel_length)
        # Normalize the kernels to have unit norm
        W = W - W.mean(dim=-1, keepdim=True)
        W = W / W.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)

        self.register_buffer("W", W)

        self.num_features = self.num_dilations * self.k * self.g * self.num_channels

    def batch(
        self,
        X: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        Apply the transformation in smaller batches.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        num_examples = X.shape[0]

        if num_examples <= batch_size:
            return self(X)

        outputs = []

        for start in range(0, num_examples, batch_size):
            end = min(start + batch_size, num_examples)
            outputs.append(self(X[start:end]))

        return torch.cat(outputs, dim=0)

    @staticmethod
    def _straight_through_assignment(
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hard argmax in the forward pass and standard softmax gradients
        in the backward pass.

        Args:
            responses:
                Tensor with shape
                [batch, groups*channels, kernels, time].

        Returns:
            Assignment tensor with the same shape.
        """

        # Differentiable soft assignment used for backpropagation.
        soft_assignment = F.softmax(
            responses,
            dim=2,
        )

        # Hard winner-take-all assignment used in the forward pass.
        max_indices = responses.argmax(
            dim=2,
            keepdim=True,
        )

        hard_assignment = torch.zeros_like(responses)
        hard_assignment.scatter_(dim=2, index=max_indices, value=1.0)

        # Forward value = hard_assignment
        # Backward gradient = gradient through soft_assignment
        return hard_assignment + soft_assignment - soft_assignment.detach()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: EEG tensor with shape [batch, channels, time].
        Returns:
            DiCT feature tensor with shape [batch, num_features].
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape [batch, channels, time], but received {tuple(X.shape)}.")

        num_examples, num_channels, _ = X.shape

        features = []

        for dilation_index in range(self.num_dilations):
            
            dilation = int(self.dilations[dilation_index].item())
            padding = int(self.paddings[dilation_index].item())
            kernels = self.W[dilation_index].to(device=X.device, dtype=X.dtype)

            # Convert from:
            # [g*k, kernel_length]
            #
            # to grouped-convolution format:
            # [channels*g*k, 1, kernel_length]
            kernels = kernels.unsqueeze(1).repeat(num_channels, 1, 1)
            responses = F.conv1d(X, kernels, dilation=dilation, padding=padding, groups=num_channels)
            responses = responses.view(num_examples, self.g * num_channels, self.k, -1)
            assignment = self._straight_through_assignment(responses)
            # Select the winning response for every group and time point.
            selected_response = (assignment * responses).sum(dim=2, keepdim=True)

            # Assign the selected response to its winning kernel
            # and aggregate across time.
            count_max = (assignment * selected_response).sum(dim=-1)

            # Signed square root avoids the dead-gradient region
            # introduced by clamp(0).
            count_max = torch.sign(count_max) * torch.sqrt(count_max.abs() + 1e-5)

            features.append(count_max)

        return torch.cat(features, dim=1).reshape(num_examples, -1)