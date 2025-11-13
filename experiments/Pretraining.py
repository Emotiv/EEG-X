

from experiments.EEG2Rep_Pretraining import EEG2Rep_Pretraining
from experiments.EEGX_Pretraining import EEGX_Pretraining
from experiments.Biot_Pretraining import Biot_Pretraining
from experiments.EEGX_Corpus_Pretraining import EEGX_Corpus_Pretraining
from training.lightning.pytorch_lightning_trainer import EEGX_Corpus_Pretraining as EEGX_Corpus_Pretraining_lightning


def Pretraining_Finetuning(config, Data):
    if config['Model'] == 'EEG2Rep':
        best_aggr_metrics_test, all_metrics = EEG2Rep_Pretraining(config, Data)
    elif config['Model'] == 'EEG-X':   
        best_aggr_metrics_test, all_metrics = EEGX_Pretraining(config, Data)
    elif config['Model'] == 'Biot':
        best_aggr_metrics_test, all_metrics = Biot_Pretraining(config, Data)
    else:
        raise ValueError("Model not implemented")   
    return best_aggr_metrics_test, all_metrics 

def Pretraining_Corpus(config, Data):
    if config['Model'] == 'EEG-X':
        if config['Framework'] == 'Pytorch':
            best_aggr_metrics_test, all_metrics = EEGX_Corpus_Pretraining(config, Data)
        if config['Framework'] == 'lightning':
            EEGX_Corpus_Pretraining_lightning(config)
    else:
        raise ValueError("Model not implemented")   
    return {}