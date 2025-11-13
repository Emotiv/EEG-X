import os
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader 
from collections import OrderedDict
from Models import utils, analysis
from experiments.train_pipeline import supervise_pipeline
from tools.utils import dataset_class
from Models.model_factory import model_factory, count_parameters
from Models.utils import load_model
from Models.loss import get_loss_module
from Models.optimizers import get_optimizer  
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from copy import deepcopy
from peft import get_peft_model, LoraConfig, TaskType


logger = logging.getLogger('__main__')


class SupervisedTrainer(object):
    def __init__( self, model, data_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        super(SupervisedTrainer, self).__init__()
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if print_conf_mat:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.model = model
        self.data_loader = data_loader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.loss_module = config['loss_module']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.data_loader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)

    def train_epoch(self, epoch_num=None):
        self.model.train()
        epoch_loss = 0
        total_batches = 0

        for i, batch in enumerate(self.data_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)

            output = self.model(X.to(self.device), labels=targets, loss_fn=self.loss_module)
            loss = output['loss']
            batch_loss = torch.sum(loss)
            total_loss = batch_loss / len(loss)  # mean loss (over samples)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            epoch_loss += total_loss.item()
            total_batches += 1

        avg_loss = epoch_loss / total_batches
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = avg_loss

        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.data_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            output = self.model(X.to(self.device), labels=targets, loss_fn=self.loss_module)
            loss = output['loss']
            batch_loss = torch.sum(loss).cpu().item()
            per_batch['targets'].append(targets.cpu().numpy())
            predictions = output['logits'].detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append(batch_loss)
            per_batch['IDs'].append(IDs)
            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss /= total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,
                                            dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        if max(targets) < 2 == 2:
            false_pos_rate, true_pos_rate, _ = roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = auc(rec, prec)

        return self.epoch_metrics, metrics_dict


class LoRA_SupervisedTrainer(object):
    def __init__( self, model, data_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        super(LoRA_SupervisedTrainer, self).__init__()
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if print_conf_mat:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.model = model
        self.data_loader = data_loader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.loss_module = config['loss_module']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']



    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.data_loader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)

    # LoRA compatible train_epoch function
    def train_epoch(self, epoch_num=None):
        self.model.train()
        epoch_loss = 0
        total_batches = 0

        for i, batch in enumerate(self.data_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)

            output = self.model.LoRA_forward(X.to(self.device), labels=targets, loss_fn=self.loss_module)
            loss = output['loss']
            batch_loss = torch.sum(loss)
            total_loss = batch_loss / len(loss)  # mean loss (over samples)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            epoch_loss += total_loss.item()
            total_batches += 1

        avg_loss = epoch_loss / total_batches
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = avg_loss

        return self.epoch_metrics

    '''
    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.data_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            total_loss = batch_loss / len(loss)  # mean loss (over samples)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.item()

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics
    '''

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.data_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            output = self.model.LoRA_forward(X.to(self.device), labels=targets, loss_fn=self.loss_module)
            loss = output['loss']
            batch_loss = torch.sum(loss).cpu().item()
            per_batch['targets'].append(targets.cpu().numpy())
            predictions = output['logits'].detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append(batch_loss)
            per_batch['IDs'].append(IDs)
            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss /= total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,
                                            dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        if max(targets) < 2 == 2:
            false_pos_rate, true_pos_rate, _ = roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = auc(rec, prec)

        return self.epoch_metrics, metrics_dict



def Supervise(config, Data):
    # -------------------------------------------- Build Model -----------------------------------------------------
    config['Data_shape'] = Data['shape']
    config['num_labels'] = Data['num_labels']
    Encoder = model_factory(config)
    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(Encoder.parameters(), lr=config['lr'], weight_decay=0)

    config['problem_type'] = 'Supervised'
    config['loss_module'] = get_loss_module()
    Encoder.to(config['device'])

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    S_trainer = SupervisedTrainer(Encoder, train_loader, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(Encoder, val_loader, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_2_model_{}.pth'.format('last'))
    supervise_pipeline(config, Encoder, S_trainer, S_val_evaluator, save_path)
    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, test_loader, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics


def Fine_Tuning(config, Data):
    # -------------------------------------------- Build Model -----------------------------------------------------
    config['Data_shape'] = Data['shape']
    config['num_labels'] = Data['num_labels']
    Encoder = model_factory(config)
    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    networks_to_optimize = [Encoder.predict_head]
    networks_not_to_optimize = [Encoder.InputEmbedding, Encoder.contex_encoder]
    # Convert parameters to tensors
    params_to_optimize = [p for net in networks_to_optimize for p in net.parameters()]
    params_not_to_optimize = [p for net in networks_not_to_optimize for p in net.parameters()]
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class([{'params': params_to_optimize, 'lr': config['lr']},
                                       {'params': params_not_to_optimize, 'lr': 0.0}])
    config['problem_type'] = 'Supervised'
    config['loss_module'] = get_loss_module()
    Encoder.to(config['device'])
    model_path = '/home/navid/emotiv-ml/emotiv-ml/fm/EEG-X/Results/Pretraining/Quant_plus_pretrain_0.75_LoRA/2025-08-05_06-02/checkpoints/Quant_plus_pretrain_0.75_LoRA_best_acc_model.pth'
    Encoder = load_model_initial(Encoder, model_path)
    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    S_trainer = SupervisedTrainer(Encoder, train_loader, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(Encoder, val_loader, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_2_model_{}.pth'.format('last'))
    supervise_pipeline(config, Encoder, S_trainer, S_val_evaluator, save_path)
    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])




    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(Encoder.parameters(), lr=config['lr'], weight_decay=0)
    config['problem_type'] = 'Supervised'
    config['loss_module'] = get_loss_module()

    S_trainer = SupervisedTrainer(best_Encoder, train_loader, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(best_Encoder, val_loader, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_3_model_{}.pth'.format('last'))
    supervise_pipeline(config, best_Encoder, S_trainer, S_val_evaluator, save_path)
    best_Encoder, optimizer, start_epoch = load_model(best_Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, test_loader, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test


def LoRA_Fine_Tuning(config, Data):
    # -------------------------------------------- Build Model -----------------------------------------------------
    config['Data_shape'] = Data['shape']
    config['num_labels'] = Data['num_labels']
    model = model_factory(config)
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    model.to(config['device'])

    # Load pretrained model (before LoRA)
    model_path = '/home/navid/emotiv-ml/emotiv-ml/fm/EEG-X/Results/Pretraining/Quant_plus_pretrain_0.75_LoRA/2025-08-05_06-02/checkpoints/Quant_plus_pretrain_0.75_LoRA_best_acc_model.pth'
    model = load_model_initial(model, model_path)

    # ----------------------------------------- Inject LoRA ---------------------------------------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------ Load Data ----------------------------------------------------------
    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config)
    # val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # ---------------------------------------- Optimizer & Training Setup -------------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['problem_type'] = 'Supervised'
    config['loss_module'] = get_loss_module()

    trainer = LoRA_SupervisedTrainer(model, train_loader, config, print_conf_mat=False)
    val_evaluator = LoRA_SupervisedTrainer(model, test_loader, config, print_conf_mat=False)


    # -------------------------------------------- Training ----------------------------------------------------------
    save_path = os.path.join(config['save_dir'], config['problem'] + '_3_model_{}.pth'.format('last'))
    supervise_pipeline(config, model, trainer, val_evaluator, save_path)

    # ------------------------------------------- Load Best Model ---------------------------------------------------
    best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_model.to(config['device'])

    # ------------------------------------------- Final Evaluation ---------------------------------------------------

    test_evaluator = LoRA_SupervisedTrainer(best_model, test_loader, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test


def load_model_initial(model, model_path, optimizer=None, resume=False, change_output=True,
               lr=None, lr_step=None, lr_factor=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    keys_to_remove = ['Layer_Norm', 'predict_head', 'PositionalEncoding', 'LocationEncoding']
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