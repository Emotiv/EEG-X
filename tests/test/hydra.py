import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from pretrain.models.hydra import build_hydra_transform


class HydraGradientTests(unittest.TestCase):
    def test_feature_loss_reaches_competing_responses(self):
        torch.manual_seed(4)
        transform = build_hydra_transform(
            "soft_channelwise", input_length=64, num_channels=2, k=8, g=2, seed=0
        )
        target = torch.randn(2, 2, 64)
        prediction = torch.randn(2, 2, 64, requires_grad=True)
        responses = []
        convolution = F.conv1d

        def capture(*args, **kwargs):
            response = convolution(*args, **kwargs)
            if response.requires_grad:
                response.retain_grad()
                responses.append(response)
            return response

        with patch.object(F, "conv1d", capture):
            with torch.no_grad():
                target_features = transform(target)
            loss = F.mse_loss(transform(prediction), target_features)
            loss.backward()
        for response in responses:
            self.assertTrue(torch.isfinite(response.grad).all())
            self.assertGreater((response.grad != 0).float().mean().item(), 0.9)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0)

    def test_seed_preserves_global_rng_and_buffers_roundtrip(self):
        state = torch.random.get_rng_state()
        transform = build_hydra_transform(
            "soft_channelwise", input_length=64, num_channels=2, g=2, seed=7
        )
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        other = build_hydra_transform(
            "soft_channelwise", input_length=64, num_channels=2, g=2, seed=8
        )
        other.load_state_dict(transform.state_dict())
        x = torch.randn(2, 2, 64)
        torch.testing.assert_close(transform(x), other(x))

    def test_flat_input_backward_is_finite(self):
        transform = build_hydra_transform(
            "soft_channelwise", input_length=64, num_channels=1, g=2, seed=0
        )
        x = torch.zeros(2, 1, 64, requires_grad=True)
        transform(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
