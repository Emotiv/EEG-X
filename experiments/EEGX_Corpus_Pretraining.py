import os 
import logging
import torch
from torch.utils.data import DataLoader 
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import OrderedDict
from Models import utils, analysis
from experiments.Supervise import SupervisedTrainer
from experiments.train_pipeline import selfsupervised_pipeline, supervise_pipeline
from tools.utils import dataset_class, dataset_class_ICA, dataset_class_Corpus
from Models.model_factory import model_factory, count_parameters
from Models.utils import load_model, save_model
from Models.loss import get_loss_module
from Models.optimizers import get_optimizer
from torch.utils.tensorboard import SummaryWriter
from tools.utils import Data_Loader
import subprocess
import matplotlib.pyplot as plt
import json

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


def EEGX_Corpus_Pretraining(config, Data):
    # ---------------------------------------- Self Supervised Data -------------------------------------
    pretrain_dataset = dataset_class_Corpus(Data['pretrain_data_list'], None, Data['pretrain_data_list_clean'], config, pretrain=True)
    pretrain_loader = DataLoader(dataset=pretrain_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)    
    # -------------------------------------------- Build Model ---------------------------------------------------------
    logger.info("Pre-Training Self Supervised model ...")
    config['Data_shape'] = Data['shape']
    config['num_labels'] = Data['num_labels']
    Encoder = model_factory(config)
    # Encoder = nn.DataParallel(Encoder)  # Enable multi-GPU support
    Encoder.to(config['device'])
    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    # Specify which networks you want to optimize
    networks_to_optimize = [Encoder.InputEmbedding, Encoder.contex_encoder, Encoder.Predictor, Encoder.Decoder]
    # Convert parameters to tensors
    params_to_optimize = [p for net in networks_to_optimize for p in net.parameters()]
    params_not_to_optimize = [p for p in Encoder.target_encoder.parameters()]
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class([{'params': params_to_optimize, 'lr': config['lr']},
                                       {'params': params_not_to_optimize, 'lr': 0.0}])
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'], config['problem'] +'_model_{}.pth'.format('last'))
    # ------------------------------------------------- Training The Model ---------------------------------------------
    logger.info('Self-Supervised training...')
    SS_trainer = EEGX_SelfSupervised_Trainer(Encoder, pretrain_loader, None, None, config, l2_reg=0, print_conf_mat=False)
    selfsupervised_pipeline(config, Encoder, SS_trainer, save_path)
    # **************************************************************************************************************** #
    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    '''
    SS_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])
    # ---------------------------------------- Linear Probing -------------------------------------------------------------
    train_repr, train_labels = make_representation(SS_Encoder, train_loader)
    test_repr, test_labels = make_representation(SS_Encoder, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    print('Test_acc:', acc_test)
    cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    print("Confusion Matrix:")
    print(cm)

    # ---------------------------------------- Fine Tuning -------------------------------------------------------------
    
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    logger.info('Starting Fine_Tuning...')
    S_trainer = SupervisedTrainer(SS_Encoder, train_loader, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(SS_Encoder, val_loader, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))
    supervise_pipeline(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, test_loader, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics
    '''
    return

def get_gpu_utilization():
    # Run nvidia-smi command to get GPU stats
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.free,memory.used,memory.total', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE
    )
    
    # Decode the result to string and split by comma
    result = result.stdout.decode('utf-8').strip().split(',')
    
    # Extract the GPU utilization and memory stats
    gpu_utilization = int(result[0].strip())  # GPU utilization percentage
    memory_free = float(result[1].strip())    # Free memory in MB
    memory_used = float(result[2].strip())    # Used memory in MB
    
    return gpu_utilization, memory_free, memory_used


def downstream_task_loader(config):
    Data = {}
    with open(config['linear_probe_paths'], 'r') as file:
        path = json.load(file)
        config['STEW_data_path'] = path['STEW_data_path']
        config['Attention_data_path'] = path['Attention_data_path']
        config['Crowdsourced_data_path'] = path['Crowdsourced_data_path']
        config['DREAMER_data_path'] = path['DREAMER_data_path']
        config['Alpha_data_path'] = path['Alpha_data_path']
        config['TUEVMultiClass_data_path'] = path['TUEVMultiClass_data_path']


        Data_npy = np.load(config['STEW_data_path'], allow_pickle=True)
        Data['train_STEW'] = np.concatenate((Data_npy['X_train'], Data_npy['X_val']))
        Data['train_label_STEW'] = np.concatenate((Data_npy['Y_train'], Data_npy['Y_val']))
        Data['test_STEW'] = Data_npy['X_test']
        Data['test_label_STEW'] = Data_npy['Y_test']

        Data_npy = np.load(config['Attention_data_path'], allow_pickle=True)
        Data['train_Attention'] = np.concatenate((Data_npy['X_train'], Data_npy['X_val']))
        Data['train_label_Attention'] = np.concatenate((Data_npy['Y_train'], Data_npy['Y_val']))
        Data['test_Attention'] = Data_npy['X_test']   
        Data['test_label_Attention'] = Data_npy['Y_test']

        Data_npy = np.load(config['Crowdsourced_data_path'], allow_pickle=True)
        Data['train_Crowdsource'] = np.concatenate((Data_npy['X_train'], Data_npy['X_val']))
        Data['train_label_Crowdsource'] = np.concatenate((Data_npy['Y_train'], Data_npy['Y_val']))
        Data['test_Crowdsource'] = Data_npy['X_test']
        Data['test_label_Crowdsource'] = Data_npy['Y_test']

        Data_npy = np.load(config['DREAMER_data_path'], allow_pickle=True)
        Data['train_DREAMER'] = np.concatenate((Data_npy['X_train'], Data_npy['X_val']))
        Data['train_label_DREAMER'] = np.concatenate((Data_npy['Y_train'], Data_npy['Y_val']))
        Data['test_DREAMER'] = Data_npy['X_test']
        Data['test_label_DREAMER'] = Data_npy['Y_test']

        Data_npy = np.load(config['Alpha_data_path'], allow_pickle=True)
        Data['train_Alpha'] = np.concatenate((Data_npy['X_train'], Data_npy['X_val']))
        Data['train_label_Alpha'] = np.concatenate((Data_npy['Y_train'], Data_npy['Y_val']))
        Data['test_Alpha'] = Data_npy['X_test']
        Data['test_label_Alpha'] = Data_npy['Y_test']

        Data_npy = np.load(config['TUEVMultiClass_data_path'], allow_pickle=True)
        Data['train_TUEV_Multi_Class'] = np.concatenate((Data_npy['X_train'], Data_npy['X_val']))
        Data['train_label_TUEV_Multi_Class'] = np.concatenate((Data_npy['Y_train'], Data_npy['Y_val']))
        Data['test_TUEV_Multi_Class'] = Data_npy['X_test']
        Data['test_label_TUEV_Multi_Class'] = Data_npy['Y_test']

    return Data

def mse_with_variance_regularization(pred, target, lambda_reg=0.1, tau=0.1):
    mse_loss = torch.mean((pred - target) ** 2)
    var_pred = torch.var(pred)
    variance_reg = torch.relu(tau - var_pred)
    loss = mse_loss + lambda_reg * variance_reg
    return loss

class EEGX_SelfSupervised_Trainer(object):
    def __init__( self, model, pre_train_loader, train_loader, test_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        super(EEGX_SelfSupervised_Trainer, self).__init__()
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if print_conf_mat:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.model = model
        self.pre_train_loader = pre_train_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.loss_module = config['loss_module']
        self.problem = config['problem']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.batch_size = config['batch_size']
        # Initialize TensorBoard writer
        self.tensorboard_writer = SummaryWriter(log_dir=config['tensorboard_dir'])
        if config['Downstream_task']:
            self.downstream_task = downstream_task_loader(config)
        self.transform = HydraMultivariateGPU(input_length=256, num_channels=14, k=8, g=32, max_num_channels=14)

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.train_loader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


    def train_epoch(self, epoch_num=None):
        self.model.copy_weight()
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.pre_train_loader):
            X, targets, X_Clean, IDs = batch
            target_rep_mask, target_prediction, X_Clean_Norm, X_reconstruct = self.model.pretrain_forward(X.to(self.device), X_Clean.to(self.device))
            align_loss = F.mse_loss(target_rep_mask, target_prediction)
            # align_loss = mse_with_variance_regularization(target_rep_mask, target_prediction)
            X_Clean_Norm_hydra = self.transform(X_Clean_Norm)
            X_reconstruct_hydra = self.transform(X_reconstruct)
            reconstruction_loss = F.mse_loss(X_Clean_Norm_hydra, X_reconstruct_hydra)
            # reconstruction_loss = torch.tensor(1e-6, requires_grad=True)
            y = self.gap(target_prediction.transpose(2, 1)).squeeze()
            y = y - y.mean(dim=0)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_y))
            cov_y = (y.T @ y) / (len(targets) - 1)
            cov_loss = off_diagonal(cov_y).pow_(2).sum().div(y.shape[-1])
            # total_loss = align_loss + std_loss + cov_loss + reconstruction_loss
            total_loss = align_loss + std_loss + cov_loss + reconstruction_loss
            # total_loss = reconstruction_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.model.momentum_update()
            total_samples += 1
            epoch_loss += total_loss.item()
        epoch_loss /= total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['Total_loss'] = epoch_loss
        self.epoch_metrics['Alignment'] = align_loss
        self.epoch_metrics['Std'] = std_loss
        self.epoch_metrics['Cov'] = cov_loss
        self.epoch_metrics['Reconstruction'] = reconstruction_loss
        # Log GPU stats using nvidia-smi
        if torch.cuda.is_available():
            gpu_utilization, memory_free, memory_used = get_gpu_utilization()
            # Log GPU memory and utilization to TensorBoard
            self.tensorboard_writer.add_scalar('GPU/Utilization', gpu_utilization, epoch_num)
            self.tensorboard_writer.add_scalar('GPU/Memory_Free_MB', memory_free, epoch_num)
            self.tensorboard_writer.add_scalar('GPU/Memory_Used_MB', memory_used, epoch_num)
        
        if (epoch_num + 1) % 5 == 0:
            self.model.eval()
            acc_test = downstream_task(self.model, self.downstream_task, self.batch_size)
            self.tensorboard_writer.add_scalar('Accuracy/STEW', acc_test['STEW'], epoch_num)
            self.tensorboard_writer.add_scalar('Accuracy/Attention', acc_test['Attention'], epoch_num)
            self.tensorboard_writer.add_scalar('Accuracy/Crowdsourced', acc_test['Crowdsourced'], epoch_num)
            self.tensorboard_writer.add_scalar('Accuracy/DREAMER', acc_test['DREAMER'], epoch_num)  
            self.tensorboard_writer.add_scalar('Accuracy/Alpha', acc_test['Alpha'], epoch_num)
            self.tensorboard_writer.add_scalar('Accuracy/TUEV_Multi_Class', acc_test['TUEV_Multi_Class'], epoch_num)
            current_avg_acc = (acc_test['STEW'] + acc_test['Attention'] + acc_test['Crowdsourced'] + acc_test['DREAMER'] + acc_test['Alpha'] + acc_test['TUEV_Multi_Class']) / 6
            self.tensorboard_writer.add_scalar('Accuracy/Average', current_avg_acc, epoch_num)
            # Initialize best accuracy if not already set
            if not hasattr(self, 'best_avg_acc'):
                self.best_avg_acc = 0.0
            # Save model if current average accuracy is better
            if current_avg_acc > self.best_avg_acc:
                self.best_avg_acc = current_avg_acc
                save_model(
                    path=os.path.join(self.save_path, 'checkpoints/' + self.problem + '_best_acc_model.pth'),
                    epoch=epoch_num,
                    model=self.model,
                    optimizer=self.optimizer
                )
            logger.info(f'New best average accuracy: {self.best_avg_acc:.4f}')
        if (epoch_num) % 100 == 0:
            save_model(
                path=os.path.join(self.save_path, 'checkpoints/' + self.problem + '_model_{}.pth'.format(epoch_num)),
                epoch=epoch_num,
                model=self.model,
                optimizer=self.optimizer
            )
        return self.epoch_metrics, self.model


import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class HydraMultivariateGPU(nn.Module):

    def __init__(self, input_length, num_channels, k = 8, g = 64, max_num_channels = 8, seed = None):

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
        
        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (15 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((15 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 15)
        W = W - W.mean(-1, keepdims = True)
        W = W / W.abs().sum(-1, keepdims = True)

        self.register_buffer("W", W)

        # self.num_features_ = self.num_dilations * self.divisor * self.k * self.h * 2
        self.num_features = self.num_dilations * self.divisor * self.k * self.h * 2

        num_channels_per = np.clip(num_channels // 2, 2, max_num_channels)
        I = torch.randint(0, num_channels, (self.num_dilations, self.divisor, self.h, num_channels_per))
        self.register_buffer("I", I)

    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):
        # make sure these tensors are on the same device
        device = X.device
        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)
        else:
            raise NotImplementedError("HydraMultivariateGPU only supports divisor = 1 for now")    

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):

                _Z = F.conv1d(X[:, self.I[dilation_index, diff_index]].sum(2).to(device) if diff_index == 0 else diff_X[:, self.I[dilation_index, diff_index]].sum(2).to(device),
                              self.W[dilation_index, diff_index].to(device), dilation=int(d), padding=int(p), groups=self.h) \
                      .view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k, device = device, dtype = max_values.dtype)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k, device = device, dtype = min_values.dtype)
                
                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z.clamp(0).sqrt()


def downstream_task(model, Data, batch_size):
    New_config = {}
    New_config['problem'] = 'STEW'
    train_dataset = dataset_class(Data['train_STEW'], Data['train_label_STEW'], New_config)
    test_dataset = dataset_class(Data['test_STEW'], Data['test_label_STEW'], New_config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_repr, train_labels = make_representation(model, train_loader)
    test_repr, test_labels = make_representation(model, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    STEW_ACC = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)

    New_config['problem'] = 'Attention'
    train_dataset = dataset_class(Data['train_Attention'], Data['train_label_Attention'], New_config)
    test_dataset = dataset_class(Data['test_Attention'], Data['test_label_Attention'], New_config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_repr, train_labels = make_representation(model, train_loader)
    test_repr, test_labels = make_representation(model, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    Attention_ACC = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    acc_test = {'STEW': STEW_ACC, 'Attention': Attention_ACC}

    New_config['problem'] = 'Crowdsourced'
    train_dataset = dataset_class(Data['train_Crowdsource'], Data['train_label_Crowdsource'], New_config)
    test_dataset = dataset_class(Data['test_Crowdsource'], Data['test_label_Crowdsource'], New_config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_repr, train_labels = make_representation(model, train_loader)
    test_repr, test_labels = make_representation(model, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    Crowdsourced_ACC = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    acc_test['Crowdsourced'] = Crowdsourced_ACC

    New_config['problem'] = 'DREAMER'
    train_dataset = dataset_class(Data['train_DREAMER'], Data['train_label_DREAMER'], New_config)
    test_dataset = dataset_class(Data['test_DREAMER'], Data['test_label_DREAMER'], New_config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_repr, train_labels = make_representation(model, train_loader)
    test_repr, test_labels = make_representation(model, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    DREAMER_ACC = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    acc_test['DREAMER'] = DREAMER_ACC

    New_config['problem'] = 'Alpha'
    train_dataset = dataset_class(Data['train_Alpha'], Data['train_label_Alpha'], New_config)
    test_dataset = dataset_class(Data['test_Alpha'], Data['test_label_Alpha'], New_config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_repr, train_labels = make_representation(model, train_loader)
    test_repr, test_labels = make_representation(model, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    Alpha_ACC = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    acc_test['Alpha'] = Alpha_ACC


    New_config['problem'] = 'TUEV_Multi_Class'
    train_dataset = dataset_class(Data['train_TUEV_Multi_Class'], Data['train_label_TUEV_Multi_Class'], New_config)
    test_dataset = dataset_class(Data['test_TUEV_Multi_Class'], Data['test_label_TUEV_Multi_Class'], New_config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_repr, train_labels = make_representation(model, train_loader)
    test_repr, test_labels = make_representation(model, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    TUEV_Multi_Class_ACC = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    acc_test['TUEV_Multi_Class'] = TUEV_Multi_Class_ACC

    return acc_test 


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def make_representation(model, data):
    out = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data):
            X, targets, IDs = batch
            rep = model.linear_prob(X.to('cuda'))
            # out_rep = torch.mean(rep, dim=1)
            out.append(rep)
            labels.append(targets)

        out = torch.cat(out, dim=0)
        labels = torch.cat(labels, dim=0)
    return out, labels


def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=3407,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe


def z_normalize_channels(x):
    """
    Z-normalize input tensor along the channel dimension.
    
    Args:
        x (numpy.ndarray): Input array of shape (sample, 14, lenght) where:
    Returns:
        numpy.ndarray: Z-normalized array of same shape as input
    """
    # Calculate mean and std along channel dimension (axis=1)
    mean = np.mean(x, axis=1, keepdims=True)  # shape: (sample, 14, 256)
    std = np.std(x, axis=1, keepdims=True)    # shape: (sample, 14, 256)
    
    # Add small epsilon to avoid division by zero
    std = std + 1e-8
    
    # Z-normalize across the 14 channels
    normalized = (x - mean) / std
    
    return normalized