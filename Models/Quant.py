import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Generate time intervals at multiple resolutions ===
def make_intervals(input_length, depth):
    max_depth = min(depth, int(np.log2(input_length)) + 1)
    intervals = []

    for n in 2 ** torch.arange(max_depth):
        indices = torch.linspace(0, input_length, n + 1).long()
        base_intervals = torch.stack((indices[:-1], indices[1:]), dim=1)
        intervals.append(base_intervals)

        # Add shifted intervals for better coverage
        if n > 1 and base_intervals.diff().median() > 1:
            shift = int(np.ceil(input_length / n / 2))
            shifted = base_intervals[:-1] + shift
            intervals.append(shifted)

    return torch.cat(intervals)

# === Compute quantile-based features within a window ===
def compute_quantiles(X, div=4):
    length = X.shape[-1]

    if length == 1:
        return X.view(X.shape[0], 1, -1)

    num_q = 1 + (length - 1) // div

    quantiles = X.quantile(torch.linspace(0, 1, num_q, device=X.device), dim=-1)
    quantiles = quantiles.permute(1, 2, 0)  # Shape: (B, C, Q)

    if num_q > 1:
        quantiles[..., 1::2] -= X.mean(-1, keepdim=True)

    return quantiles

# === Interval model for one representation (e.g., raw, diff, freq, etc.) ===
class IntervalModel:
    def __init__(self, input_length, depth=4, div=4):
        self.div = div
        self.intervals = make_intervals(input_length, depth)

    def transform(self, X):
        features = []
        for start, end in self.intervals:
            segment = X[..., start:end]
            features.append(compute_quantiles(segment, div=self.div))
        return torch.cat(features, dim=-1)


class SharedConvPerChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(1, 1, kernel_size=5, padding=2, stride=2)
        self.conv7 = nn.Conv1d(1, 1, kernel_size=7, padding=3, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)  # reduce length by ~3

    def forward(self, x):
        B, C, L = x.shape
        x = x.view(B * C, 1, L)

        out3 = F.relu(self.conv3(x))  # (B*C,1,L/2)
        out5 = F.relu(self.conv5(x))  # (B*C,1,L/2)
        out7 = F.relu(self.conv7(x))  # (B*C,1,L/2)

        out = torch.cat([out3, out5, out7], dim=1)  # (B*C,3,L/2)
        # Apply max pooling on last dimension (length)
        out = self.pool(out)  # (B*C, 3, L/6) roughly

        out = out.view(B, C, -1)  # (B, C, 3 * L/6) == (B, C, L/2)

        return out

# === Full quantizer using multiple representations ===
class Quant_plus(nn.Module):
    def __init__(self, depth=4, div=4):
        super().__init__()
        self.depth = depth
        self.div = div
        self.models = {}
        self.fitted = False
        self.shared_conv = SharedConvPerChannel()

        # Define multiple time series views
        self.representations = [
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),  # Smoothed diff
            lambda X: X.diff(n=2),  # Second derivative
            lambda X: torch.fft.rfft(X).abs(),  # Frequency magnitude
        ]

    def fit_transform(self, X):
        features = []
        for i, rep_fn in enumerate(self.representations):
            Z = rep_fn(X)
            self.models[i] = IntervalModel(Z.shape[-1], self.depth, self.div)
            features.append(self.models[i].transform(Z))

        quantile_features = torch.cat(features, dim=-1)
        cnn_features = self.shared_conv(X)
        self.fitted = True
        return torch.cat([quantile_features, cnn_features], dim=-1)

    def transform(self, X):
        assert self.fitted, "Call fit_transform first."
        features = []
        for i, rep_fn in enumerate(self.representations):
            Z = rep_fn(X)
            features.append(self.models[i].transform(Z))

        quantile_features = torch.cat(features, dim=-1)
        cnn_features = self.shared_conv(X)
        return torch.cat([quantile_features, cnn_features], dim=-1)


class Quant(nn.Module):
    def __init__(self, depth=4, div=4):
        super().__init__()
        self.depth = depth
        self.div = div
        self.models = {}
        self.fitted = False

        # Define multiple time series views
        self.representations = [
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),  # Smoothed diff
            lambda X: X.diff(n=2),  # Second derivative
            lambda X: torch.fft.rfft(X).abs(),  # Frequency magnitude
        ]

    def fit_transform(self, X):
        features = []
        for i, rep_fn in enumerate(self.representations):
            Z = rep_fn(X)
            self.models[i] = IntervalModel(Z.shape[-1], self.depth, self.div)
            features.append(self.models[i].transform(Z))

        quantile_features = torch.cat(features, dim=-1)
        self.fitted = True
        return quantile_features

    def transform(self, X):
        assert self.fitted, "Call fit_transform first."
        features = []
        for i, rep_fn in enumerate(self.representations):
            Z = rep_fn(X)
            features.append(self.models[i].transform(Z))

        quantile_features = torch.cat(features, dim=-1)
        return quantile_features

# === Run a simple test ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.randn(10, 2, 128).to(device)  # Move input to device

    quant = Quant(depth=4, div=4).to(device)  # Move model to device
    features = quant.fit_transform(X)

    print("Extracted feature shape:", features.shape)
