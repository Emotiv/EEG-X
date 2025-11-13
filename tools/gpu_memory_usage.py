import torch
import time
import psutil
import os
import subprocess
import json
import sys
from copy import deepcopy
import argparse
# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the project root to sys.path
sys.path.append(project_root)
from Models.model_factory import model_factory


parser = argparse.ArgumentParser()

def get_memory_usage():
    """Returns RAM usage and GPU memory usage in GB for all GPUs."""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    
    gpu_memory = {
        f"GPU {i}": {
            "Allocated": torch.cuda.memory_allocated(i) / (1024 ** 3),
            "Reserved": torch.cuda.memory_reserved(i) / (1024 ** 3),
        }
        for i in range(torch.cuda.device_count())
    }

    return ram_usage, gpu_memory

def measure_inference_time(model, input_tensor, repetitions=100):
    """Measures average inference time per sample in seconds."""
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(repetitions):
            _ = model.linear_prob(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
    
    avg_time = (end_time - start_time) / repetitions
    return avg_time


def load_model(model, model_path, optimizer=None, resume=False, change_output=True,
               lr=None, lr_step=None, lr_factor=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only = True)
    state_dict = deepcopy(checkpoint['state_dict'])
    keys_to_remove = ['Layer_Norm', 'predict_head', 'PositionalEncoding']
    if change_output:
        for key, val in checkpoint['state_dict'].items():
            if any(key.startswith(k) for k in keys_to_remove):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model

def get_model_parameters(model):
    """Returns the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def get_gpu_info():
    """Returns GPU model and the number of GPUs available."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                            stdout=subprocess.PIPE, text=True)
    gpu_models = result.stdout.strip().split('\n')
    num_gpus = len(gpu_models)
    return gpu_models, num_gpus

def get_gpu_usage():
    """Returns GPU utilization percentage for all GPUs as a list."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE, text=True)
    
    # Convert output to a list of integers (one per GPU)
    gpu_utilization = [int(value.strip()) for value in result.stdout.strip().split('\n')]
    
    return gpu_utilization


def report_model_performance(model, input_shape, repetitions=100):
    """Generates a report on memory usage, inference speed, and GPU utilization."""
    input_tensor = torch.randn(*input_shape).cuda()
    
    ram, _ = get_memory_usage()
    inference_time = measure_inference_time(model, input_tensor, repetitions)
    gpu_mem_after, _ = get_memory_usage()
    gpu_usage = get_gpu_usage()
    total_params = get_model_parameters(model)
    gpu_models, num_gpus = get_gpu_info()
    
    print(f"GPU Models: {gpu_models}")
    print(f"Model is running on device: {next(model.parameters()).device}")
    print(f"RAM Usage: {ram:.2f} GB")
    print(f"GPU Memory Usage: {gpu_mem_after:.2f} GB")
    for i, usage in enumerate(gpu_usage):
        print(f"GPU {i}: {usage}% utilization")
    print(f"Inference Speed per 1000 Sample: {inference_time * 1000:.3f} ms")
    print(f"Total Trainable Parameters: {total_params}")
    return total_params, inference_time, gpu_mem_after 


# ----------------------------------------- Parameters and Hyperparameters ----------------------------------------------
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout regularization ratio')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--val_interval', type=int, default=5, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy'}, default='loss', help='Metric used for best epoch')
parser.add_argument('--ICA', default=True, type=bool, choices={'True', 'False'}, help='Having ICA reconstruction in loss function or Not')
# -------------------------------------------------- EEG-X ----------------------------------------------------------
parser.add_argument('--Model', default='EEG-X', choices={'EEG-X','EEG2Rep', 'Biot', 'EEGPT', 'LaBraM', 'MAEEG'}, help="Model Type")
parser.add_argument('--Training_mode', default='Linear_Prob', choices={'Supervise_training', 'Pretraining_Finetuning', 'Linear_Prob'}, 
                    help="Training Mode")
parser.add_argument('--Evaluation', default='In-domain', choices={'In-domain', 'Cross-domain'}, help="Evaluation Mode")
parser.add_argument('--Input_Embedding', default='Channel-wise', choices={'Channel-wise', 'CNN'}, help="Input Embedding Architecture")
parser.add_argument('--T_Pos_Encoding', default=['Sin'], choices={'Sin', 'Vector_Embed'}, help="Temporal Position Encoding Method")
parser.add_argument('--C_Pos_Encoding', default=['Location'], choices={'Location', 'Vector_Embed'}, help="Channel info Encoding Method")

parser.add_argument('--layers', type=int, default=4, help="Number of layers for the context/target encoders")
parser.add_argument('--pre_layers', type=int, default=2, help="Number of layers for the Predictor")
parser.add_argument('--mask_ratio', type=float, default=0.5, help=" masking ratio")
parser.add_argument('--momentum', type=float, default=0.99, help="Beta coefficient for EMA update")

parser.add_argument('--patch_size', type=int, default=128, help='Patch size for data segmentation. Data is preprocessed to 128Hz')
parser.add_argument('--patch_stride', type=int, default=32, help='Patch stride for data segmentation')
parser.add_argument('--emb_size', type=int, default=1024, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of feedforward network of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
config = args.__dict__
torch.cuda.set_device(1)  # Set to GPU 1
config['device'] = 'cuda'
config['num_labels'] = 2
embedding_dims = [16, 128]
input_lengths = [256, 512, 1024]
results = []

for emb_size in embedding_dims:
    config['emb_size'] = emb_size
    for input_length in input_lengths:
        config['Data_shape'] = (1000, 14, input_length)
        Encoder = model_factory(config)
        Encoder.to(config['device'])
        
        total_params, inference_time, gpu_mem_after = report_model_performance(Encoder, config['Data_shape'])
        results.append((emb_size, input_length, inference_time, gpu_mem_after))

import matplotlib.pyplot as plt

# Plotting the results
embedding_dims = [result[0] for result in results]
input_lengths = [result[1] for result in results]
inference_times = [result[2] for result in results]
gpu_memory_usages = [result[3] for result in results]

fig, ax = plt.subplots(figsize=(12, 8))
# Plotting as a scatter plot
scatter = ax.scatter(input_lengths, inference_times, s=[(length * emb_size) / 10 for length, emb_size in zip(input_lengths, embedding_dims)], c=embedding_dims, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)

# Annotating each point with the embedding size
for i, txt in enumerate(embedding_dims):
    ax.annotate(txt, (input_lengths[i], inference_times[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding a color bar to indicate embedding sizes
cbar = plt.colorbar(scatter)
cbar.set_label('Embedding Size')

ax.set_xlabel('Input Length')
ax.set_ylabel('Inference Time for 1000 samples (seconds)')
ax.set_title('Inference Time vs Input Length')
plt.savefig('inference_time_vs_input_length2.png')
plt.show()

# Plotting GPU memory usage
fig, ax = plt.subplots(figsize=(12, 8))
# Plotting as a scatter plot
scatter = ax.scatter(input_lengths, gpu_memory_usages, s=[(length * emb_size) / 10 for length, emb_size in zip(input_lengths, embedding_dims)], c=embedding_dims, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)

# Annotating each point with the embedding size
for i, txt in enumerate(embedding_dims):
    ax.annotate(txt, (input_lengths[i], gpu_memory_usages[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding a color bar to indicate embedding sizes
cbar = plt.colorbar(scatter)
cbar.set_label('Embedding Size')

ax.set_xlabel('Input Length')
ax.set_ylabel('GPU Memory Usage (GB)')
ax.set_title('GPU Memory Usage vs Input Length')
plt.savefig('gpu_memory_usage_vs_input_length2.png')
plt.show()


'''
config['Data_shape'] = (1000, 14, 256)
emb_sizes = [16, 64, 128, 256, 512, 1024]
results = []

for emb_size in emb_sizes:
    config['emb_size'] = emb_size
    Encoder = model_factory(config)
    Encoder.to(config['device'])
    
    total_params, inference_time, gpu_mem_after = report_model_performance(Encoder, config['Data_shape'])
    results.append((emb_size, total_params, inference_time, gpu_mem_after))

import matplotlib.pyplot as plt

# Plotting the results
embedding_sizes = [result[0] for result in results]
total_parameters = [result[1] for result in results]
inference_times = [result[2] for result in results]
gpu_memory_usages = [result[3] for result in results]

fig, ax = plt.subplots(figsize=(12, 8))
# Plotting as a scatter plot where the size of the point is based on embedding size
scatter = ax.scatter(total_parameters, inference_times, s=[size * 10 for size in embedding_sizes], c=embedding_sizes, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)

# Annotating each point with the embedding size
for i, txt in enumerate(embedding_sizes):
    ax.annotate(txt, (total_parameters[i], inference_times[i]), textcoords="offset points", xytext=(0,-10), ha='center')

# Adding a color bar to indicate embedding sizes
cbar = plt.colorbar(scatter)
cbar.set_label('Embedding Size')

# Formatting the x-axis to show values in millions and in log scale
ax.set_xscale('log')
ax.set_xticks(total_parameters)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xticklabels([f'{int(x/1e3)}K' for x in total_parameters])

ax.set_xlabel('Total Parameters (log scale)')
ax.set_ylabel('Inference Time for 1000 samples (seconds)')
ax.set_title('Inference Time vs Total Parameters with Embedding Sizes')
plt.savefig('inference_time_vs_parameters.png')
plt.show()

# Plotting GPU memory usage
fig, ax = plt.subplots(figsize=(12, 8))
# Plotting as a scatter plot where the size of the point is based on embedding size
scatter = ax.scatter(total_parameters, gpu_memory_usages, s=[size * 10 for size in embedding_sizes], c=embedding_sizes, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)

# Annotating each point with the embedding size
for i, txt in enumerate(embedding_sizes):
    ax.annotate(txt, (total_parameters[i], gpu_memory_usages[i]), textcoords="offset points", xytext=(0,-10), ha='center')

# Adding a color bar to indicate embedding sizes
cbar = plt.colorbar(scatter)
cbar.set_label('Embedding Size')

# Formatting the x-axis to show values in millions and in log scale
ax.set_xscale('log')
ax.set_xticks(total_parameters)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xticklabels([f'{int(x/1e3)}K' for x in total_parameters])

ax.set_xlabel('Total Parameters (log scale)')
ax.set_ylabel('GPU Memory Usage (GB)')
ax.set_title('GPU Memory Usage vs Total Parameters with Embedding Sizes')
plt.savefig('gpu_memory_usage_vs_parameters.png')
plt.show()
'''