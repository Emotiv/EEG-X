import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeClassifierCV
from tools.utils import dataset_class
from sklearn.metrics import accuracy_score, confusion_matrix



def Hydra(Data):
    # -------------------------------------------- Build Model -----------------------------------------------------
    transform = HydraMultivariateGPU(input_length=256, num_channels=14, k=8, g=128, max_num_channels=14)
    # --------------------------------- Load Data -------------------------------------------------------------
    
    # Data['All_train_data'] = torch.fft.fft(torch.tensor(Data['All_train_data']), dim=2)

    # Split data into chunks to avoid 32-bit indexing limits
    chunk_size = 1000  # Adjust based on available memory
    train_chunks = []
    
    for i in range(0, len(Data['All_train_data']), chunk_size):
        chunk = torch.tensor(Data['All_train_data'][i:i+chunk_size]).float().to('cuda')
        train_chunks.append(transform(chunk))
    
    X_training_transform = torch.cat(train_chunks, dim=0)
    
    test_data = torch.tensor(Data['test_data']).float().to('cuda')
    X_test_transform = transform(test_data)
    X_test_transform = transform(test_data)

    scaler = SparseScaler()

    X_training_transform = scaler.fit_transform(X_training_transform)
    X_test_transform = scaler.transform(X_test_transform)

    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    classifier.fit(X_training_transform.cpu(), Data['All_train_label'])

    predictions = classifier.predict(X_test_transform.cpu())
    accuracy = accuracy_score (Data['test_label'],predictions)
    conf_matrix = confusion_matrix(Data['test_label'],predictions)
    print(accuracy)
    print(conf_matrix)

    return 


class HydraMultivariateGPU(nn.Module):

    def __init__(self, input_length, num_channels, k = 8, g = 64, max_num_channels = 8, seed = None):

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
        torch.device('cuda')
        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9)
        W = W - W.mean(-1, keepdims = True)
        W = W / W.abs().sum(-1, keepdims = True)

        self.register_buffer("W", W)

        # self.num_features_ = self.num_dilations * self.divisor * self.k * self.h * 2
        self.num_features = self.num_dilations * self.divisor * self.k * self.h * 2

        num_channels_per = np.clip(num_channels // 2, 2, max_num_channels)
        self.I = torch.randint(0, num_channels, (self.num_dilations, self.divisor, self.h, num_channels_per))

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

        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):

                _Z = F.conv1d(X[:, self.I[dilation_index, diff_index]].sum(2).to('cuda') if diff_index == 0 else diff_X[:, self.I[dilation_index, diff_index]].sum(2).to('cuda'),
                              self.W[dilation_index, diff_index].to('cuda'), dilation = d, padding = p,
                              groups = self.h) \
                      .view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k, device = X.device)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k, device = X.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z.clamp(0).sqrt()


class SparseScaler():

    def __init__(self, mask = True, exponent = 4):

        self.mask = mask
        self.exponent = exponent

        self.fitted = False

    def fit(self, X):

        assert not self.fitted, "Already fitted."

        X = X.clamp(0).sqrt()

        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon

        self.fitted = True

    def transform(self, X):

        assert self.fitted, "Not fitted."

        X = X.clamp(0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)