"""Count nominal dense arithmetic in EEG-X without allocating GPU memory."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def self_attention_block(tokens, width):
    # Q/K/V/output projections and the expansion/contraction FFN, then QK and AV.
    return 24 * tokens * width**2 + 4 * tokens**2 * width


def cross_attention_block(visible, hidden, width):
    return (20 * hidden + 4 * visible) * width**2 + 4 * hidden * visible * width


def dense_counts(model, channels, samples, classes):
    """One window, with one multiply-add counted as two nominal FLOPs.

    This scope excludes FFT, bias additions, elementwise operations, norms,
    nonlinearities, reductions, memory movement and optimizer updates. It counts
    logical unpadded products, independent of the GPU kernel's instruction mix.
    """
    contract = model.contract
    patch = contract['patch_size']
    if (min(channels, samples, classes) < 1 or samples % patch
            or samples != model.input_size or contract['patch_stride'] != patch
            or contract['mask_ratio'] != 0.75):
        raise ValueError('Expected the current nonoverlapping, masked EEG-X geometry')
    objective = contract['objective']
    if objective is not None and objective['name'] != 'raw_soft_dict_v2':
        raise ValueError('Dense training counts support raw MSE plus SoftDiCT only')
    width, depth = contract['width'], contract['depth']
    tokens = channels * (samples // patch)
    visible = tokens // 4
    if visible < 1:
        raise ValueError('The encoder requires visible patches')
    hidden = tokens - visible
    tokenizer = 2 * tokens * width * (
        model.encoder.raw_projection.in_features + model.encoder.spectral_projection.in_features)
    inference = tokenizer + depth * self_attention_block(tokens, width) + 2 * width * classes
    reconstruction = (tokenizer + depth * self_attention_block(visible, width)
        + contract['predictor_depth'] * cross_attention_block(visible, hidden, width)
        + contract['decoder_depth'] * self_attention_block(tokens, width)
        + 2 * tokens * width * patch)
    # Other dense layers need forward, input gradients and weight gradients.
    # Tokenizer inputs are fixed EEG/STFT, so only its weight gradients are needed.
    core_training = 3 * reconstruction - tokenizer
    dictionary_forward = 0
    if objective:
        dictionary_forward = (2 * channels * samples * objective['groups'] * objective['k']
                              * len(objective['dilations']) * sum(objective['kernel_sizes']))
    # Target forward, prediction forward, backward response recomputation and
    # prediction input gradient. Fixed dictionary filters receive no gradients.
    dictionary_training = 4 * dictionary_forward
    return {'inference': inference, 'pretraining_forward_core': reconstruction,
            'pretraining_forward_backward_core': core_training,
            'soft_dict_training': dictionary_training,
            'pretraining': core_training + dictionary_training}


def verify(model, eeg, ids, classes):
    """Check the formulas against actual CPU reference forward/backward execution."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from pretrain.models.factory import Classifier, masks

    expected = dense_counts(model, eeg.shape[1], eeg.shape[2], classes)
    classifier = Classifier(model.encoder, classes).eval()
    with FlopCounterMode(display=False) as count, torch.no_grad():
        classifier(eeg, ids)
    observed_inference = count.get_total_flops()
    model.train()
    mask = masks([f'flop-audit-{i}' for i in range(len(eeg))], ids,
                 samples=eeg.shape[2], seed=241)
    with FlopCounterMode(display=False) as count:
        model(eeg, ids, mask=mask)['loss'].backward()
    observed_training = count.get_total_flops()
    if (observed_inference != len(eeg) * expected['inference']
            or observed_training != len(eeg) * expected['pretraining']):
        raise ValueError('Analytical counts differ from the reference operator trace')
    model.zero_grad(set_to_none=True)
    return {'inference': observed_inference, 'pretraining': observed_training,
            'training_operators': {str(k): v for k, v in count.get_flop_counts()['Global'].items()}}


def report(widths, datasets, *, trace=False):
    import torch
    from pretrain.models.factory import create_model
    from pretrain.pipeline.dataset import DatasetReference, read_metadata
    from pretrain.pipeline.checkpoint import dataset_identity

    torch.set_num_threads(2)
    protocol = json.loads((ROOT / 'data/protocol.json').read_text())
    rows = []
    for dataset in datasets:
        split = protocol['datasets'][dataset.replace('-', '_')]['split_seed']
        metadata = read_metadata(DatasetReference(ROOT / 'data' / f'{dataset}.h5', split))
        channels, samples, classes = (len(metadata['channels']), metadata['window_samples'],
                                      len(metadata['classes']))
        for width in widths:
            torch.manual_seed(241)
            model = create_model({'width': width, 'depth': 8, 'heads': 8,
                                  'input_samples': samples, 'seed': 241,
                                  'normalization_backend': 'torch', 'dictionary_backend': 'torch'})
            counts = dense_counts(model, channels, samples, classes)
            row = {'dataset': dataset, 'size': f'd8w{width}', 'channels': channels,
                   'samples': samples, 'classes': classes, 'train_windows': metadata['counts']['train'],
                   'dataset_identity': dataset_identity(metadata),
                   'counts': counts, 'inference_gflops': counts['inference'] / 1e9,
                   'training_tflops_per_epoch': counts['pretraining'] * metadata['counts']['train'] / 1e12}
            if trace:
                row['reference_trace'] = verify(model, torch.randn(1, channels, samples),
                                                torch.tensor(metadata['channel_ids'])[None], classes)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {'scope': 'nominal dense operations; MAC=2; forward/backward with dictionary recomputation',
            'excluded': ['FFT', 'bias additions', 'normalization', 'activations', 'softmax',
                         'elementwise arithmetic', 'reductions', 'optimizer', 'data movement'],
            'torch': str(torch.__version__),
            'formula_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--widths', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--datasets', nargs='+', default=['stew', 'dreamer-arousal', 'tuev-binary'])
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.widths, args.datasets, trace=args.verify)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
