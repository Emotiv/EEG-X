import os 
import logging
import torch
from torch.utils.data import DataLoader 
from torch import nn
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
from tools.utils import dataset_class, dataset_class_ICA
from Models.model_factory import model_factory, count_parameters
from Models.utils import load_model
from Models.loss import get_loss_module
from Models.optimizers import get_optimizer
from torch.utils.tensorboard import SummaryWriter
import subprocess


logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


def EEGX_Pretraining(config, Data):
    # ---------------------------------------- Self Supervised Data loader -------------------------------------------
    pre_train_dataset = dataset_class_ICA(Data['All_train_data'], Data['All_train_label'], Data['train_data_clean'], config)
    pre_train_loader = DataLoader(dataset=pre_train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # -------------------------------------------- Build Model ---------------------------------------------------------
    logger.info("Pre-Training Self Supervised model ...")
    config['Data_shape'] = Data['shape']
    config['num_labels'] = Data['num_labels']
    Encoder = model_factory(config)
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
    save_path = os.path.join(config['save_dir'], config['problem'] +'model_{}.pth'.format('last'))
    Encoder.to(config['device'])
    '''
    Encoder = nn.DataParallel(Encoder)  # Enable multi-GPU support
    Encoder.to(config['device'])
    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    # Specify which networks you want to optimize
    networks_to_optimize = [Encoder.module.InputEmbedding, Encoder.module.contex_encoder, Encoder.module.Predictor, Encoder.module.Decoder]
    # Convert parameters to tensors
    params_to_optimize = [p for net in networks_to_optimize for p in net.parameters()]
    params_not_to_optimize = [p for p in Encoder.module.target_encoder.parameters()]
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class([{'params': params_to_optimize, 'lr': config['lr']},
                                       {'params': params_not_to_optimize, 'lr': 0.0}])
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'], config['problem'] +'model_{}.pth'.format('last'))'
    '''
    # ------------------------------------------------- Training The Model ---------------------------------------------
    logger.info('Self-Supervised training...')
    SS_trainer = EEGX_SelfSupervised_Trainer(Encoder, pre_train_loader, train_loader, test_loader, config, l2_reg=0, print_conf_mat=False)
    selfsupervised_pipeline(config, Encoder, SS_trainer, save_path)
    # **************************************************************************************************************** #
    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
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
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Initialize TensorBoard writer
        self.tensorboard_writer = SummaryWriter(log_dir=config['tensorboard_dir'])

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
            reconstruction_loss = F.mse_loss(X_Clean_Norm, X_reconstruct)
            # reconstruction_loss = torch.tensor(1e-6, requires_grad=True)
            y = self.gap(target_prediction.transpose(2, 1)).squeeze()
            y = y - y.mean(dim=0)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_y))
            cov_y = (y.T @ y) / (len(targets) - 1)
            cov_loss = off_diagonal(cov_y).pow_(2).sum().div(y.shape[-1])
            total_loss = align_loss + std_loss + cov_loss + reconstruction_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.model.momentum_update()

            total_samples += 1
            epoch_loss += total_loss.item()
        epoch_loss /= total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        self.epoch_metrics['align'] = align_loss
        self.epoch_metrics['std'] = std_loss
        self.epoch_metrics['cov'] = cov_loss
        self.epoch_metrics['rec'] = reconstruction_loss
        # Log GPU stats using nvidia-smi
        if torch.cuda.is_available():
            gpu_utilization, memory_free, memory_used = get_gpu_utilization()
            # Log GPU memory and utilization to TensorBoard
            self.tensorboard_writer.add_scalar('GPU/Utilization', gpu_utilization, epoch_num)
            self.tensorboard_writer.add_scalar('GPU/Memory_Free_MB', memory_free, epoch_num)
            self.tensorboard_writer.add_scalar('GPU/Memory_Used_MB', memory_used, epoch_num)

        if (epoch_num + 1) % 5 == 0:
            self.model.eval()
            train_repr, train_labels = make_representation(self.model, self.train_loader)
            test_repr, test_labels = make_representation(self.model, self.test_loader)
            clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            y_hat = clf.predict(test_repr.cpu().detach().numpy())
            acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
            print('Test_acc:', acc_test)
            result_file = open(self.save_path + '/linear_result.txt', 'a+')
            print(f'{int(epoch_num)}, {acc_test}, {align_loss}, {std_loss}, {cov_loss}, {reconstruction_loss}', file=result_file)
            result_file.close()

        return self.epoch_metrics, self.model
    

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