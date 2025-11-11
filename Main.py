import argparse
import logging
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import seed_everything

from experiments.LinearProb import Linear_Prob, Train_prediction_head
from experiments.Pretraining import Pretraining_Corpus, Pretraining_Finetuning
from experiments.Supervise import Fine_Tuning, LoRA_Fine_Tuning, Supervise
from tools.utils import Data_Loader, Setup, print_title

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _str2bool(value: str) -> bool:
    """Convert common string representations of truth to boolean."""
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # System
    parser.add_argument('--gpu', type=int, default=0, help='GPU index, -1 for CPU')
    parser.add_argument('--console', action='store_true', help='Optimize printout for console output; otherwise for file')
    parser.add_argument('--seed', type=int, default=1234, help='Seed used for splitting sets')

    # I/O
    parser.add_argument('--data_dir', default='/data/EEG-X/Processed/Kalunga2016', help='Data directory')
    parser.add_argument('--output_dir', default='fm/EEG-X/Results', help='Root output directory. Time-stamped directories will be created inside.')
    parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
    parser.add_argument('--pretrain_data_write_path', default='/work/corpus/subcorpus/epoc/corpus-subset{}-{}.db', help='Pretrain data path')
    parser.add_argument('--pretrain_data_read_path', default='/data/binary_files_by_subjects/epoc', help='Pretrain data path')
    parser.add_argument('--dictionary_path', default='/work/corpus/subcorpus/epoc', help='Pretrain data path')
    parser.add_argument('--linear_probe_paths', default='fm/EEG-X/datasets/linear_probe_paths.json', help='JSON file with paths for linear probing')

    # Parameters and hyperparameters
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='Number of pre-training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=100, help='Number of fine-tuning or supervise epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout regularization ratio')
    parser.add_argument('--Norm', type=_str2bool, default=False, help='Data normalization flag')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Proportion of the train-set to be used as validation')
    parser.add_argument('--val_interval', type=int, default=5, help='Evaluate on validation every N epochs (>=1)')
    parser.add_argument('--key_metric', choices=['loss', 'accuracy'], default='loss', help='Metric used for best epoch')
    parser.add_argument('--ICA', type=_str2bool, default=True, help='Whether to include ICA reconstruction in the loss function')

    # EEG-X
    parser.add_argument('--Model', default='EEG-X', choices=['EEG-X', 'EEG2Rep', 'Biot', 'EEGPT', 'LBraM'], help='Model type')
    parser.add_argument('--Framework', default='Pytorch', choices=['lightning', 'Ray', 'Pytorch'], help='Training framework')
    parser.add_argument('--Training_mode', default='Supervise', choices=['Supervise', 'Fine_Tuning', 'LoRA_Fine_Tuning','Linear_Prob', 'Pretraining_finetuning'], help='Training mode')
    parser.add_argument('--Input_Embedding', default='Quant', choices=['CNN', 'FFT', 'Quant'], help='Input embedding architecture')
    parser.add_argument('--Evaluation', default='In-domain', choices=['In-domain', 'Cross-domain'], help='Evaluation mode')
    parser.add_argument('--T_Pos_Encoding', default=['Sin'], choices=['Sin', 'Vector_Embed'], help='Temporal position encoding method')
    parser.add_argument('--C_Pos_Encoding', default=['Location'], choices=['Location', 'Vector_Embed'], help='Channel encoding method')
    parser.add_argument('--layers', type=int, default=4, help='Number of layers for the context/target encoders')
    parser.add_argument('--pre_layers', type=int, default=2, help='Number of layers for the predictor')
    parser.add_argument('--Decoder_layers', type=int, default=2, help='Number of layers for the decoder')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Masking ratio')
    parser.add_argument('--momentum', type=float, default=0.99, help='Beta coefficient for EMA update')

    # Data segmentation
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for data segmentation. Data is preprocessed to 128Hz')
    parser.add_argument('--patch_stride', type=int, default=32, help='Patch stride for data segmentation')
    parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
    parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of feedforward network of transformer layer')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')

    # Corpus I/O
    parser.add_argument('--data_provider', default='local', choices=['corpus', 'local'])
    parser.add_argument('--Downstream_task', type=_str2bool, default=True, help='Enable downstream task')
    parser.add_argument('--corpus_subset', type=int, default=5, choices=[1, 2, 3, 4, 5], help='Corpus subset')
    parser.add_argument('--hertz', type=int, default=128, help='Sampling rate')
    parser.add_argument('--channels', type=int, default=14, help='Number of channels')
    parser.add_argument('--seconds', type=int, default=2, help='Window length in seconds')
    parser.add_argument('--stride', type=int, default=64, help='Stride used for windowing')
    parser.add_argument('--corpus_input_path', type=str, default='/data/binary_files_by_subjects', help='Corpus input path')
    parser.add_argument('--number_dataloader_worker', type=int, default=8, help='Number of workers used to load data')
    parser.add_argument('--pretrain_dataset_percent', type=float, default=1.0, help='Percentage of the dataset used for pretraining')
    parser.add_argument('--probe_frequency', type=int, default=5, help='Frequency to run linear probe')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Accelerator to use')
    parser.add_argument('--running_on_gcp', type=_str2bool, default=False, help='Flag indicating execution on GCP')
    parser.add_argument('--predefined_file_list', type=str, default='', help='Predefined file list for training, if any')

    return parser


def set_seed(framework: str, seed: int) -> None:
    if framework == 'lightning':
        seed_everything(seed, workers=True)
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    mode = config['Training_mode']

    if mode == 'Linear_Prob':
        return Linear_Prob(config)

    data = Data_Loader(config)
    if mode == 'Supervise':
        return Supervise(config, data)
    if mode == 'Pretraining_finetuning':
        return Pretraining_Finetuning(config, data)
    if mode == 'Fine_Tuning':
        return Fine_Tuning(config, data)
    if mode == 'LoRA_Fine_Tuning':
        return LoRA_Fine_Tuning(config, data)

    raise ValueError(f"Unsupported training mode: {mode}")


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    summary = 'Best Model Test Summary: '
    summary += ' | '.join(f'{key}: {value}' for key, value in metrics.items())
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.Framework, args.seed)

    config = Setup(args)
    print_title(config['problem'])
    logger.info('Loading Data ...')

    best_metrics = run_training(config)

    print_title(config['problem'])
    print(summarize_metrics(best_metrics))


if __name__ == '__main__':
    main()