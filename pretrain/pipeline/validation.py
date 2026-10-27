"""Fixed-mask reconstruction and auxiliary validation metrics."""

import math

import torch

from pretrain.compute import autocast_context
from pretrain.models.factory import masks
from pretrain.pipeline.checkpoint import stream_seed
from pretrain.pipeline.data import evaluation_batches


def measure(model, view, config, device, budget):
    """Keep the selection MSE reduction unchanged and measure the auxiliary loss."""
    numerator, count = 0.0, 0
    auxiliary_sum, samples = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for batch in evaluation_batches(view, config, device):
            budget.check()
            mask = masks(batch['window_id'], batch['channel_ids'], samples=config['input_samples'],
                         seed=stream_seed(config['seed'], 'validation_mask')).to(device)
            eeg, ids = batch['eeg'].to(device), batch['channel_ids'].to(device)
            with autocast_context(device, config['precision']):
                result = model(eeg, ids, mask=mask)
                # Preserve the previous masked-patch numerator and denominator exactly.
                errors = model.patch_errors(eeg, result['pred'])
            batch_numerator = float((errors * mask).sum())
            auxiliary = float(result['hydra_loss'])
            if not math.isfinite(batch_numerator) or not math.isfinite(auxiliary):
                raise ValueError('Nonfinite pretraining validation loss')
            numerator += batch_numerator
            count += int(mask.sum())
            auxiliary_sum += len(eeg) * auxiliary
            samples += len(eeg)
    if not count:
        raise ValueError('Validation contains no masked patches')
    name = model.auxiliary_metric.removeprefix('train/')
    return {'raw_mse': numerator / count, name: auxiliary_sum / samples}
