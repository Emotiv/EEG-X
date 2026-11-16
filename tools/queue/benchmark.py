"""Measure resident-input classifier inference under the current shared GPU load."""

import statistics


def inference(model, eeg, ids, config, *, warmup=10, iterations=50):
    import torch
    from pretrain.compute import autocast_context, configure_compute

    model.eval()
    batches = {}
    with (configure_compute(config, eeg.device) as policy, torch.inference_mode(),
          autocast_context(eeg.device, config['precision'])):
        for count in (1, min(64, len(eeg))):
            x, channels = eeg[:count], ids[:count]
            for _ in range(warmup):
                model(x, channels)
            torch.cuda.synchronize()
            events = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x, channels)
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            values = [start.elapsed_time(end) for start, end in events]
            median = statistics.median(values)
            batches[str(count)] = {'median_ms': median, 'samples_per_second': count * 1000 / median}
    return {'scope': 'classifier forward with resident EEG; shared GPU load; no data loading',
            'weights': 'random initialization after three supervised updates',
            'precision': config['precision'], 'compute': policy,
            'warmup': warmup, 'iterations': iterations, 'batches': batches}
