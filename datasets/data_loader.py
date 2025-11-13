import os
import numpy as np
import logging
from sklearn import model_selection
import sys

from axon.ml import Corpus
from axon.ml import Dict
import hashlib


logger = logging.getLogger(__name__)

def load_old(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/' + config['problem'] + '_Filtered.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + config['problem'] + '_Filtered.npy', allow_pickle=True)
        if np.any(Data_npy['val_data']):
            Data['train_data'] = Data_npy['train_data']
            Data['train_label'] = Data_npy['train_label']
            Data['val_data'] = Data_npy['val_data']
            Data['val_label'] = Data_npy['val_label']
            Data['test_data'] = Data_npy['test_data']
            Data['test_label'] = Data_npy['test_label']
            Data['max_len'] = Data['train_data'].shape[2]
            Data['All_train_data'] = np.concatenate((Data['train_data'], Data['val_data']))
            Data['All_train_label'] = np.concatenate((Data['train_label'], Data['val_label']))
            if config['Evaluation'] == 'Cross-domain':
                Data['pre_train_data'], Data['pre_train_label'] = Cross_Domain_loader(Data_npy)
                logger.info(
                    "{} samples will be used for self-supervised Pre_training".format(len(Data['pre_train_label'])))
        else:
            Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = \
                split_dataset(Data_npy['train_data'], Data_npy['train_label'], 0.1)
            Data['All_train_data'] = Data_npy['train_data']
            Data['All_train_label'] = Data_npy['train_label']
            Data['test_data'] = Data_npy['test_data']
            Data['test_label'] = Data_npy['test_label']
            Data['max_len'] = Data['train_data'].shape[2]
        Data['shape'] = Data['train_data'].shape
        Data['num_labels'] = int(max(Data['train_label']))+1
    if config['Model'] == 'EEG-X':
        clean_npy = np.load(config['data_dir'] + '/' + config['problem'] + '_Clean.npy', allow_pickle=True)
        Data['train_data_clean'] = np.concatenate((clean_npy['train_data'], clean_npy['val_data']))

    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def load(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/' + config['problem'] + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + config['problem'] + '.npy', allow_pickle=True)
        # Select only channels 4 and 9 (0-based indexing: channels 3 and 8)
        Data['train_data'] = Data_npy['X_train'][:, [4, 9], :]
        Data['train_label'] = np.where(np.isin(Data_npy['Y_train'], [1, 2, 3]), 0, 1)
        # Data['train_label'] = Data_npy['Y_train']
        Data['val_data'] = Data_npy['X_val'][:, [4, 9], :]
        Data['val_label'] = np.where(np.isin(Data_npy['Y_val'], [1, 2, 3]), 0, 1)
        # Data['val_label'] = Data_npy['Y_val']
        Data['test_data'] = Data_npy['X_test'][:, [4, 9], :]
        Data['test_label'] = np.where(np.isin(Data_npy['Y_test'], [1, 2, 3]), 0, 1)
        # Data['test_label'] = Data_npy['Y_test']
        Data['All_train_data'] = np.concatenate((Data['train_data'], Data['val_data']))
        Data['All_train_label'] = np.concatenate((Data['train_label'], Data['val_label']))
        Data['shape'] = Data['train_data'].shape
        Data['num_labels'] = int(max(Data['train_label']))+1
    if config['Model'] == 'EEG-X':
        if 'X_train_clean' in Data_npy and 'X_val_clean' in Data_npy:
            Data['train_data_clean'] = np.concatenate((Data_npy['X_train_clean'], Data_npy['X_val_clean']))
        else:
            Data['train_data_clean'] = np.concatenate((Data_npy['X_train'], Data_npy['X_val']))

    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def load_TUEV(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/TUEV_train.npy'):
        logger.info("Loading preprocessed data ...")
        Train_data = np.load(config['data_dir'] + '/TUEV_train.npy', allow_pickle=True)
        Val_data = np.load(config['data_dir'] + '/TUEV_val.npy', allow_pickle=True)
        Test_data = np.load(config['data_dir'] + '/TUEV_test.npy', allow_pickle=True)

        Data['train_data'] = Train_data['X_train']
        Data['train_label'] = Train_data['y_train']
        Data['val_data'] = Val_data['X_val']
        Data['val_label'] = Val_data['y_val']
        Data['test_data'] = Test_data['X_test']
        Data['test_label'] = Test_data['y_test']
        Data['max_len'] = Data['train_data'].shape[2]
        Data['shape'] = Data['train_data'].shape
        Data['num_labels'] = int(max(Data['train_label'])) + 1
        Data['All_train_data'] = np.concatenate((Data['train_data'], Data['val_data']))
        Data['All_train_label'] = np.concatenate((Data['train_label'], Data['val_label']))
        if config['Evaluation'] == 'Cross-domain':
            Data['pre_train_data'], Data['pre_train_label'] = Cross_Domain_loader(Data)
            logger.info(
                    "{} samples will be used for self-supervised Pre_training".format(len(Data['pre_train_label'])))
    if config['Model'] == 'EEG-X':
        if 'X_train_clean' in Train_data and 'X_val_clean' in Val_data:
            Data['train_data_clean'] = np.concatenate((Train_data['X_train_clean'], Val_data['X_val_clean']))
        else:
            Data['train_data_clean'] = np.concatenate((Train_data['X_train'], Val_data['X_val']))
    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))
    return Data


def load_TUAB(config):
    Data = {}
    data_path = config['data_dir'] + '/processed_128hz_1280seqlen/windowed_10S'
    Data['val_data'] = os.path.join(data_path, 'val')
    Data['val_label'] = []
    Data['train_data'] = os.path.join(data_path, 'train')
    Data['train_data_clean'] = os.path.join(data_path, 'train')
    Data['train_label'] = []
    Data['test_data'] = os.path.join(data_path, 'test')
    Data['test_label'] = []
    Data['All_train_data'] = [Data['train_data'], Data['val_data']]
    Data['All_train_clean'] = [Data['train_data'], Data['val_data']]
    Data['All_train_label'] = []
    Data['shape'] = [334852,19,1280]
    Data['num_labels'] = 2
    return Data


def load_npy(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/' + config['problem'] + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + config['problem'] + '.npy', allow_pickle=True)

        if np.any(Data_npy.item().get('val_data')):
            Data['train_data'] = Data_npy.item().get('train_data')
            Data['train_label'] = Data_npy.item().get('train_label')
            Data['val_data'] = Data_npy.item().get('val_data')
            Data['val_label'] = Data_npy.item().get('val_label')
            Data['All_train_data'] = Data_npy.item().get('All_train_data')
            Data['All_train_label'] = Data_npy.item().get('All_train_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[1]
        else:
            Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = \
                split_dataset(Data_npy.item().get('train_data'), Data_npy.item().get('train_label'), 0.1)
            Data['All_train_data'] = Data_npy.item().get('train_data')
            Data['All_train_label'] = Data_npy.item().get('train_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[2]

    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def Cross_Domain_loader(domain_data):
    All_train_data = domain_data.item().get('All_train_data')
    All_train_label = domain_data.item().get('All_train_label')
    # Load DREAMER for Pre-Training
    DREAMER = np.load('Dataset/DREAMER/DREAMER.npy', allow_pickle=True)
    All_train_data = np.concatenate((All_train_data, DREAMER.item().get('All_train_data')), axis=0)
    All_train_label = np.concatenate((All_train_label, DREAMER.item().get('All_train_label')), axis=0)

    # Load Crowdsource for Pre-Training
    Crowdsource = np.load('Dataset/Crowdsource/Crowdsource.npy', allow_pickle=True)
    All_train_data = np.concatenate((All_train_data, Crowdsource.item().get('All_train_data')), axis=0)
    All_train_label = np.concatenate((All_train_label, Crowdsource.item().get('All_train_label')), axis=0)
    return All_train_data, All_train_label


def split_dataset(data, label, validation_ratio):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label


def pretrain_corpus(config):
    # Function to build the corpus from raw unlabled data or load an existing corpus
    # Corpus can be used as a numpy matrix but it's in fact a binary tree. See corpus.py.
    corpus_config = Dict(
        {
        'write_path'     : config['pretrain_data_write_path'], 
        'read_path'      : config['pretrain_data_read_path'], #config['pretrain_data_path'], # test on '/data/sessions-tiny', then train on '/data/sessions-small',
        'distribution'   : [1.0, 0.0, 0.0],
        'batch_size'     : config['batch_size'], # 256 
        'hz'             : config['hertz'], # 128
        'channels'       : config['channels'], # 14
        'seconds'        : config['seconds'], # 2
        'stride'         : config['stride'], # 64
        'num_samples'    : int(config['seconds'] * config['hertz']),
        'is_torch'       : True
        })
    
    def get_signature():
        directory_path = config['dictionary_path']
        subset_str = str(config['corpus_subset'])
        
        pattern = f"corpus-subset{subset_str}-"
        for filename in os.listdir(directory_path):
            if filename.startswith(pattern):
                signature = filename[len(pattern):-3]
                return signature
                
        raise ValueError(f"No file found for corpus subset {config['corpus_subset']}")

    # config['dim_ff'] = int(config['seconds'] * config['hertz'])

    cc = corpus_config
    # state = ( cc.write_path, cc.read_path, tuple(cc.distribution), cc.batch_size,
    #             cc.hz, cc.channels, cc.seconds, cc.stride, cc.is_torch)

    # state_bytes = str(state).encode('utf-8')
    # state_object = hashlib.sha256(state_bytes)
    # cc.signature = state_object.hexdigest()
    fname = cc.write_path.format(config['corpus_subset'], get_signature())    
    
    print('fname', fname)
    # print('signature', cc.signature)
    # print(state_bytes)
    try:
        corpus = Corpus.load(fname)
        print(corpus)
    except:
        corpus = Corpus(
            path=cc.read_path, 
            recLen=int(cc.channels * cc.hz * cc.seconds), 
            skipLen=cc.channels * cc.stride,
            numSamples=cc.num_samples,
            targets=cc.distribution, 
            batchSize=cc.batch_size, 
            isTorch=cc.is_torch
        )
        corpus.save(fname)
    
    Data = Dict()
    Data['pretrain_data_list'] = corpus.data
    Data['pretrain_data_list_clean'] = corpus.data2
    Data['shape'] = [1, config['channels'],config['hertz']*config['seconds']]
    Data['num_labels'] = 2
    return Data

def load_corpus(config):
    Data = {}
    Data['pretrain_data_list'] = np.load(config['data_dir'] + '/Sub5.npy')
    Data['shape'] = [1, config['channels'],config['hertz']*config['seconds']]
    Data['num_labels'] = 2
    return Data

def load_corpus(config):
    Data = {}
    Data['pretrain_data_list'] = np.load(config['data_dir'] + '/Sub5.npy')
    Data['shape'] = [1, config['channels'],config['hertz']*config['seconds']]
    Data['num_labels'] = 2
    return Data