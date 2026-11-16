"""Validation diagnostics must preserve raw-MSE selection and random state."""

import torch
from torch.nn import functional as F

from pretrain.models.factory import create_model, masks
from pretrain.pipeline.budget import NoGpuBudget
from pretrain.pipeline.checkpoint import stream_seed
from pretrain.pipeline.data import Windows, evaluation_batches
from pretrain.pipeline.validation import measure
from tests.test.pipeline.data import prepared_fixture


def test_soft_dict_validation_matches_raw_baseline_and_weighted_auxiliary(tmp_path):
    config = {'input_samples': 256, 'seed': 241, 'depth': 1, 'width': 16,
              'precision': 'fp32', 'microbatch': 3, 'workers': 0,
              'objective': {'name': 'raw_soft_dict_v2', 'weight': 1.0, 'temperature': 1.0,
                            'groups': 2, 'k': 2, 'kernel_sizes': [3], 'dilations': [1]}}
    view = Windows(prepared_fixture(tmp_path / 'data'), 'validation')
    assert len(view) % config['microbatch'] != 0
    model = create_model(config).eval()
    rng = torch.get_rng_state().clone()
    raw_sum, patches, auxiliary_sum, count = 0.0, 0, 0.0, 0
    with torch.inference_mode():
        for batch in evaluation_batches(view, config, torch.device('cpu')):
            eeg, ids = batch['eeg'], batch['channel_ids']
            mask = masks(batch['window_id'], ids, samples=256,
                         seed=stream_seed(241, 'validation_mask'))
            prediction, mask = model.predict_masked(eeg, ids, mask=mask)
            # This is the previous validation reduction, before auxiliary diagnostics.
            raw_sum += float((model.patch_errors(eeg, prediction) * mask).sum())
            patches += int(mask.sum())
            size, channels, samples = eeg.shape
            reconstruction = prediction.reshape(size, samples // 32, channels, 32)
            reconstruction = reconstruction.permute(0, 2, 1, 3).reshape(size * channels, 1, samples)
            target = model.hydra(eeg.reshape(size * channels, 1, samples))
            auxiliary_sum += size * float(F.mse_loss(model.hydra(reconstruction), target))
            count += size
    values = measure(model, view, config, torch.device('cpu'), NoGpuBudget())
    assert values['raw_mse'] == raw_sum / patches
    assert values['soft_dict'] == auxiliary_sum / count
    assert torch.equal(rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())
