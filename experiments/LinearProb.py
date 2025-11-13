import torch
import json
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader 
from tools.utils import dataset_class
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from Models.model_factory import model_factory
from collections import OrderedDict
from sklearn.metrics import f1_score
from training.lightning.eegx_module import EEGXLightningModule
import yaml
import cupy as cp
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.preprocessing import StandardScaler as cuStandardScaler
from Models.loss import get_loss_module
from Models.optimizers import get_optimizer 
import pytorch_lightning as pl


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import json
from collections import OrderedDict


class Prediction_head_trainer(pl.LightningModule):
    def __init__(self, encoder, lr, num_labels):
        super().__init__()
        self.encoder = encoder
        self.loss_module = get_loss_module()
        self.lr = lr
        self.num_labels = num_labels

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        X, targets, _ = batch
        X, targets = X.to(self.device), targets.to(self.device)
        predictions = self.encoder.model(X)
        loss = self.loss_module(predictions, targets)
        batch_loss = torch.sum(loss)
        total_loss = batch_loss / len(loss)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        X, targets, _ = batch
        X, targets = X.to(self.device), targets.to(self.device)
        predictions = self.encoder.model(X)
        preds = torch.argmax(predictions, dim=1)
        acc = (preds == targets).float().mean()
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_acc": acc}

    def configure_optimizers(self):
        # Freeze encoder except for prediction head
        networks_to_optimize = [self.encoder.model.predict_head]
        networks_frozen = [self.encoder.model.InputEmbedding, self.encoder.model.contex_encoder]

        params_to_optimize = [p for net in networks_to_optimize for p in net.parameters()]
        frozen_params = [p for net in networks_frozen for p in net.parameters()]

        optimizer_class = get_optimizer("RAdam")
        optimizer = optimizer_class([
            {'params': params_to_optimize, 'lr': self.lr},
            {'params': frozen_params, 'lr': 0.0}
        ])
        return optimizer


def Train_prediction_head(config):
    epoch_metrics = OrderedDict()

    # Load paths to model and config
    with open(config['linear_probe_paths'], 'r') as file:
        paths = json.load(file)
        config['Linear_prob_model'] = paths['Linear_prob_model']
        config['Linear_prob_config'] = paths['Linear_prob_config']
        config['device'] = 'cuda'

    # Load data
    Data = {}
    data_npy = np.load('/data/EEG-X/Processed/Attention/Attention.npy', allow_pickle=True)
    Data['All_train_data'] = np.concatenate((data_npy['X_train'], data_npy['X_val']))
    Data['All_train_label'] = np.concatenate((data_npy['Y_train'], data_npy['Y_val']))
    Data['test_data'] = data_npy['X_test']
    Data['test_label'] = data_npy['Y_test']

    config['num_labels'] = int(max(Data['All_train_label'])) + 1
    config['Data_shape'] = Data['All_train_data'].shape

    # Load pre-trained encoder
    with open(config['Linear_prob_config'], 'r') as f:
        hyperparams = yaml.safe_load(f)
        pretrain_config = hyperparams.get('config', {})

    encoder = EEGXLightningModule.load_from_checkpoint(config['Linear_prob_model'], config=pretrain_config)
    encoder.to(config['device'])

    # Create datasets and loaders
    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    # Set up Lightning module and trainer
    linear_probe_module = Prediction_head_trainer(encoder, config['lr'], config['num_labels'])

    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', config['finetune_epochs']),
        accelerator='gpu' if config['device'] == 'cuda' else 'cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False  # I'll manually save
    )

    # Train
    trainer.fit(linear_probe_module, train_loader)

    # Save the full model checkpoint
    trainer.save_checkpoint("EEG-X_prediction_head.ckpt")
    print("✅ Saved full model as 'EEG-X_prediction_head.ckpt'")

    # Test
    test_results = trainer.test(linear_probe_module, test_loader, verbose=False)
    if test_results and 'test_acc' in test_results[0]:
        acc = test_results[0]['test_acc']
        print(f"Test Accuracy: {acc:.4f}")
        epoch_metrics['test_acc'] = acc

    return epoch_metrics



def Linear_Prob(config):
    epoch_metrics = OrderedDict()
    with open(config['linear_probe_paths'], 'r') as file:
        path = json.load(file)
        config['Linear_prob_model'] = path['Linear_prob_model']
        config['Linear_prob_config'] = path['Linear_prob_config']
        config['device'] = 'cuda'
        # Create list to store results
        results = []
        
        for key, value in path.items():
            if key.endswith('data_path'):
                Downstream_task = key.split('_')[0]
                print(Downstream_task)
                Data = {}
                config[key] = value
                Data_npy = np.load(value, allow_pickle=True)
                Data['All_train_data'] = np.concatenate((Data_npy['X_train'], Data_npy['X_val']))
                Data['All_train_label'] = np.concatenate((Data_npy['Y_train'], Data_npy['Y_val'])).astype(np.int32)
                Data['test_data'] = Data_npy['X_test']
                Data['test_label'] = Data_npy['Y_test'].astype(np.int32)
                config['num_labels'] = int(max(Data['All_train_label'])) + 1
                config['Data_shape'] = Data['All_train_data'].shape
                model = model_factory(config)
                if config['Framework'] == 'Pytorch':
                    Encoder = load_model(model, config['Linear_prob_model'])
                    #  Encoder = model
                else:
                    with open(config['Linear_prob_config'], 'r') as f:
                        hyperparameter_ = yaml.safe_load(f)
                        pretrain_config = hyperparameter_.get('config', {})
                    Encoder = EEGXLightningModule.load_from_checkpoint(config['Linear_prob_model'], config=pretrain_config)
                    # Encoder = EEGXLightningModule(config)
                    # Encoder = load_model_lightning(model, config['Linear_prob_model'])

                Encoder.to(config['device'])
                train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config)
                test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

                train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

                train_rep, train_labels = make_representation(Encoder, train_loader, config['Framework'])
                test_rep, test_labels = make_representation(Encoder, test_loader, config['Framework'])

                # Convert tensors to numpy arrays if necessary, here assumed to be in numpy format
                train_rep = train_rep.cpu().detach().numpy()
                train_labels = train_labels.cpu().detach().numpy()
                test_rep = test_rep.cpu().detach().numpy()
                test_labels = test_labels.cpu().detach().numpy()
                # Create logistic regression model and Fit model on training data
                clf = fit_lr(train_rep, train_labels)
                clf.fit(train_rep, train_labels)
                # Predict on test data
                y_hat = clf.predict(test_rep)
                y_prob = clf.predict_proba(test_rep)[:, 1]  # Get the probabilities for the positive class
                # Calculate accuracy
                acc_test = accuracy_score(test_labels, y_hat)
                # Calculate AUROC
                if len(set(test_labels)) > 2:
                    auroc_test = f1_score(test_labels, y_hat, average='weighted')
                else:
                    auroc_test = roc_auc_score(test_labels, y_prob)
                print('Test_acc:', acc_test)
                cm = confusion_matrix(test_labels, y_hat)
                print("Confusion Matrix:")
                print(cm)
                epoch_metrics['Accuracy'] = acc_test
                epoch_metrics['AUROC'] = auroc_test
                
                # Store results for this problem
                results.append({
                    'Problem': Downstream_task,
                    'Accuracy': acc_test,
                    'AUROC': auroc_test
                })
        
        # Save results to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(f"{config['pred_dir']}/linear_probe_results.csv", index=False)

    return epoch_metrics


def downstream_task_loader(config):
    Data = {}
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

    return Data
'''

def Zero_Shot(Data):
    epoch_metrics = OrderedDict()
    # Path to the JSON file
    json_file_path = 'Checkpoints/TUAB/TUAB_configuration.json'
    # Read and parse the JSON file
    with open(json_file_path, 'r') as file:
        config = json.load(file)
    config['device'] = 'cuda'

    config['num_labels'] = int(max(Data['Y'])) + 1
    config['Data_shape'] = Data['X'].shape
    model = model_factory(config)
    Encoder = load_model(model, 'Checkpoints/TUAB/TUABmodel_last.pth')
    Encoder.to(config['device'])
    unique_ids = set(Data['id'])

    accuracies = []
    aurocs = []
    for subject_id in unique_ids:
        print(f"Training with subject {subject_id} left out.")
        # Split the data into training and testing sets based on the subject ID
        train_indices = [i for i, id in enumerate(Data['id']) if id != subject_id]
        test_indices = [i for i, id in enumerate(Data['id']) if id == subject_id]

        train_data = Data['X'][train_indices]
        train_labels = Data['Y'][train_indices]
        test_data = Data['X'][test_indices]
        test_labels = Data['Y'][test_indices]

        train_dataset = dataset_class(train_data, train_labels, config)
        test_dataset = dataset_class(test_data, test_labels, config)

        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

        train_rep, train_labels = make_representation(Encoder, train_loader)
        test_rep, test_labels = make_representation(Encoder, test_loader)

        # Convert tensors to numpy arrays if necessary, here assumed to be in numpy format
        train_rep = train_rep.cpu().detach().numpy()
        train_labels = train_labels.cpu().detach().numpy()
        test_rep = test_rep.cpu().detach().numpy()
        test_labels = test_labels.cpu().detach().numpy()

        # Create logistic regression model and Fit model on training data
        clf = fit_lr(train_rep, train_labels)
        clf.fit(train_rep, train_labels)

        # Predict on test data
        y_hat = clf.predict(test_rep)
        y_prob = clf.predict_proba(test_rep)[:, 1]  # Get the probabilities for the positive class

        # Calculate accuracy
        acc_test = accuracy_score(test_labels, y_hat)

        # Calculate AUROC
        if len(set(test_labels)) > 2:
            auroc_test = f1_score(test_labels, y_hat, average='weighted')
        else:
            auroc_test = roc_auc_score(test_labels, y_prob)
        
        print(f'Test_acc for subject {subject_id}:', acc_test)
        cm = confusion_matrix(test_labels, y_hat)
        print("Confusion Matrix:")
        print(cm)

        epoch_metrics[f'Accuracy_subject_{subject_id}'] = acc_test
        epoch_metrics[f'AUROC_subject_{subject_id}'] = auroc_test
        accuracies.append(acc_test)
        aurocs.append(auroc_test)
    # Calculate the average accuracy over all subjects
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_aurocs = sum(aurocs) / len(aurocs)

    print(f'Average accuracy over all subjects: {avg_accuracy}')
    print(f'Average AUROC over all subjects: {avg_aurocs}')
    epoch_metrics['Average_Accuracy'] = avg_accuracy
    epoch_metrics['Average_AUROC'] = avg_aurocs

    return epoch_metrics
'''

def load_model_lightning(model, model_path, change_output=True):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])


    # Remove 'model._orig_mod.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if k.startswith('model._orig_mod.'):
            new_key = k.replace('model._orig_mod.', 'model.')
        elif k.startswith('model.'):
            new_key = k  # fine
        else:
            continue  # skip if it doesn't match expected format
        new_state_dict[new_key] = v

    # Optional: remove layers you don’t want to load
    if change_output:
        keys_to_remove = ['model.Layer_Norm', 'model.predict_head', 'model.PositionalEncoding', 'model.LocationEncoding']
        for key in list(new_state_dict.keys()):
            if any(key.startswith(k) for k in keys_to_remove):
                new_state_dict.pop(key)

        # Load model weights
        model.load_state_dict(new_state_dict, strict=False)
        # model.load_state_dict(state_dict, strict=False)

    # Optional: Display useful info
    epoch = checkpoint.get('epoch', 'N/A')
    print(f'Loaded model from {model_path}. Epoch: {epoch}')
    if 'hyper_parameters' in checkpoint:
        print(f"Loaded hyperparameters: {checkpoint['hyper_parameters']}")
    if 'callbacks' in checkpoint:
        print(f"Loaded with callbacks: {[c.__class__.__name__ for c in checkpoint['callbacks']]}")

    return model

def load_model(model, model_path, optimizer=None, resume=False, change_output=True,
               lr=None, lr_step=None, lr_factor=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    # Remove unwanted prefix from keys
    new_state_dict = {}
    for key, val in state_dict.items():
        new_key = key.replace("base_model.model.", "")
        new_state_dict[new_key] = val

    keys_to_remove = ['Layer_Norm', 'PositionalEncoding', 'InputEmbedding.Norm', 'predict_head']
    if change_output:
        for key in list(new_state_dict.keys()):
            if any(key.startswith(k) for k in keys_to_remove):
                new_state_dict.pop(key)

    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded model from {model_path}. Epoch: {checkpoint.get('epoch', 'N/A')}")

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

def make_representation(model, data, framework):
    out = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data):
            X, targets, IDs = batch
            if framework == 'Pytorch':
                rep = model.linear_prob(X.to('cuda'))
            else:
                rep = model.model.linear_prob(X.to('cuda'))
            # out_rep = torch.mean(rep, dim=1)
            out.append(rep)
            labels.append(targets)

        out = torch.cat(out, dim=0)
        labels = torch.cat(labels, dim=0)
    return out, labels

def fit_lr_gpu(features, y):
    # Convert to GPU arrays if not already
    if not isinstance(features, cp.ndarray):
        features = cp.asarray(features)
    if not isinstance(y, cp.ndarray):
        y = cp.asarray(y)
    model = cuLogisticRegression(
        fit_intercept=True,
        C=0.1
    )
    scaler = cuStandardScaler()
    features = scaler.fit_transform(features)
    model.fit(features, y)

    return model, scaler  

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
            multi_class='ovr',
            class_weight='balanced'
        )
    )
    pipe.fit(features, y)
    return pipe