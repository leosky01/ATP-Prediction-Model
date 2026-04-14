"""
EnhancedMLP neural network and dataset utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import HIDDEN_SIZES, DROPOUT_RATES, LEAKY_RELU_SLOPE


class TennisDataset(Dataset):
    """Simple (X, y) dataset for tennis match features."""

    def __init__(self, X, y, indices):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EnhancedMLP(nn.Module):
    """
    Multi-layer perceptron with BatchNorm, LeakyReLU and Dropout.

    Architecture:  input → BN → (Linear → LeakyReLU → Dropout → BN)* → Linear(1)
    """

    def __init__(self, input_dim: int,
                 hidden_sizes=None, dropout_rates=None, slope=None):
        super().__init__()
        hidden_sizes = hidden_sizes or HIDDEN_SIZES
        dropout_rates = dropout_rates or DROPOUT_RATES
        slope = slope if slope is not None else LEAKY_RELU_SLOPE

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        in_dim = input_dim
        for i, (h, d) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(slope))
            layers.append(nn.Dropout(d))
            # BatchNorm on all hidden layers except the last
            if i < len(hidden_sizes) - 1:
                layers.append(nn.BatchNorm1d(h))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.input_bn(x))
