"""
DurationPredictor – Match duration regression with uncertainty estimation.

V2 Architecture: StackedResidualSE encoder + 2 DeepHead (mean, logvar).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..config import (
    DURATION_ENCODER_DIMS, DURATION_ENCODER_BLOCKS, DURATION_ENCODER_DROPOUT,
    DURATION_HEAD_HIDDEN, DURATION_HEAD_BLOCKS, DURATION_HEAD_DROPOUT,
    SE_REDUCTION,
)
from .blocks import StackedResidualSE, DeepHead


class DurationDataset(Dataset):
    """Dataset for duration prediction: (X, y_minutes)."""

    def __init__(self, X, y, indices):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DurationPredictor(nn.Module):
    """
    V2 Regression MLP with Gaussian NLL output.

    Encoder: StackedResidualSE
    Heads: DeepHead for mean and log_var
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list | None = None,
        encoder_blocks: int | None = None,
        encoder_dropout: float | None = None,
        head_hidden: int | None = None,
        head_blocks: int | None = None,
        head_dropout: float | None = None,
    ):
        super().__init__()
        encoder_dims = encoder_dims or DURATION_ENCODER_DIMS
        encoder_blocks = encoder_blocks if encoder_blocks is not None else DURATION_ENCODER_BLOCKS
        encoder_dropout = encoder_dropout if encoder_dropout is not None else DURATION_ENCODER_DROPOUT
        head_hidden = head_hidden if head_hidden is not None else DURATION_HEAD_HIDDEN
        head_blocks = head_blocks if head_blocks is not None else DURATION_HEAD_BLOCKS
        head_dropout = head_dropout if head_dropout is not None else DURATION_HEAD_DROPOUT

        self.encoder = StackedResidualSE(
            input_dim=input_dim,
            layer_dims=encoder_dims,
            dropout=encoder_dropout,
            se_reduction=SE_REDUCTION,
            n_blocks_per_layer=encoder_blocks,
        )
        enc_out = encoder_dims[-1]

        self.mean_head = DeepHead(enc_out, 1, head_hidden, head_blocks, head_dropout)
        self.logvar_head = DeepHead(enc_out, 1, head_hidden, head_blocks, head_dropout)

    def forward(self, x):
        features = self.encoder(x)
        mean = self.mean_head(features)
        log_var = self.logvar_head(features)
        return mean, log_var


class GaussianNLLLoss(nn.Module):
    """Gaussian NLL: 0.5 * (log_var + (target - mean)^2 / exp(log_var))."""

    def forward(self, mean, log_var, target):
        log_var = torch.clamp(log_var, min=-10, max=10)
        variance = torch.exp(log_var)
        nll = 0.5 * (log_var + (target - mean) ** 2 / variance)
        return nll.mean()
