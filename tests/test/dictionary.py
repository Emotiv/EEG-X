import copy

import pytest
import torch
import torch.nn.functional as F

from pretrain.models.dictionary import SoftDiCT
from pretrain.objectives import DEFAULT_HYDRA, validate_objective


def test_soft_counts_preserve_dictionary_competition_and_temporal_sums():
    transform = SoftDiCT(DEFAULT_HYDRA)
    samples = 64
    features = transform(torch.randn(2, 3, samples)).reshape(2, 3, 10, 8, 32)
    torch.testing.assert_close(features.sum(-1), torch.full((2, 3, 10, 8), float(samples)))
    assert not list(transform.parameters())
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("flat", [False, True])
def test_dictionary_loss_backpropagates_through_reconstruction(flat):
    transform = SoftDiCT({**DEFAULT_HYDRA, "groups": 2, "k": 4})
    target = torch.randn(2, 2, 64)
    prediction = torch.zeros_like(target) if flat else torch.randn_like(target)
    prediction.requires_grad_()
    loss = F.mse_loss(transform(prediction), transform(target).detach())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_dictionary_seed_does_not_consume_training_rng_and_buffers_round_trip():
    rng = torch.get_rng_state()
    left = SoftDiCT({**DEFAULT_HYDRA, "seed": 241})
    assert torch.equal(rng, torch.get_rng_state())
    right = SoftDiCT({**DEFAULT_HYDRA, "seed": 242})
    assert not torch.equal(left.filters_9, right.filters_9)
    right.load_state_dict(left.state_dict())
    eeg = torch.randn(1, 2, 64)
    torch.testing.assert_close(left(eeg), right(eeg), rtol=0, atol=0)


@pytest.mark.parametrize(
    "key,value", [("kernel_sizes", [8]), ("dilations", [0]), ("dilations", [1, 1]), ("k", 1)]
)
def test_dictionary_rejects_invalid_support(key, value):
    objective = copy.deepcopy(DEFAULT_HYDRA)
    objective[key] = value
    with pytest.raises(ValueError):
        validate_objective(objective)
