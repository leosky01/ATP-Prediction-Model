"""
GamesPredictor – Over/Under total games classification + exact games regression.

V2 Architecture: StackedResidualSE encoder + DeepHead heads.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..config import (
    GAMES_ENCODER_DIMS, GAMES_ENCODER_BLOCKS, GAMES_ENCODER_DROPOUT,
    GAMES_HEAD_HIDDEN, GAMES_HEAD_BLOCKS, GAMES_HEAD_DROPOUT,
    SE_REDUCTION,
)
from .blocks import StackedResidualSE, DeepHead


class GamesDataset(Dataset):
    """Dataset for games prediction: (X, y_ou, y_total)."""

    def __init__(self, X, y_ou, y_total, indices):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y_ou = torch.tensor(y_ou[indices], dtype=torch.float32).unsqueeze(1)
        self.y_total = torch.tensor(y_total[indices], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y_ou)

    def __getitem__(self, idx):
        return self.X[idx], self.y_ou[idx], self.y_total[idx]


class GamesPredictor(nn.Module):
    """
    V2 Multi-task model: StackedResidualSE encoder + 2 DeepHead outputs.

    Head 1: Over/Under classification (logit)
    Head 2: Total games regression
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
        encoder_dims = encoder_dims or GAMES_ENCODER_DIMS
        encoder_blocks = encoder_blocks if encoder_blocks is not None else GAMES_ENCODER_BLOCKS
        encoder_dropout = encoder_dropout if encoder_dropout is not None else GAMES_ENCODER_DROPOUT
        head_hidden = head_hidden if head_hidden is not None else GAMES_HEAD_HIDDEN
        head_blocks = head_blocks if head_blocks is not None else GAMES_HEAD_BLOCKS
        head_dropout = head_dropout if head_dropout is not None else GAMES_HEAD_DROPOUT

        # Encoder
        self.encoder = StackedResidualSE(
            input_dim=input_dim,
            layer_dims=encoder_dims,
            dropout=encoder_dropout,
            se_reduction=SE_REDUCTION,
            n_blocks_per_layer=encoder_blocks,
        )
        enc_out = encoder_dims[-1]

        # Heads
        self.ou_head = DeepHead(enc_out, 1, head_hidden, head_blocks, head_dropout)
        self.total_head = DeepHead(enc_out, 1, head_hidden, head_blocks, head_dropout)

    def forward(self, x):
        features = self.encoder(x)
        ou_logit = self.ou_head(features)
        total_pred = self.total_head(features)
        return ou_logit, total_pred


class GamesLoss(nn.Module):
    """Combined BCE + MSE loss for multi-task games prediction."""

    def __init__(self, bce_weight: float = 0.7, mse_weight: float = 0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, ou_logit, total_pred, y_ou, y_total):
        bce_loss = self.bce(ou_logit, y_ou)
        # Only compute MSE on non-NaN targets
        valid = ~torch.isnan(y_total.squeeze())
        if valid.any():
            mse_loss = self.mse(total_pred[valid], y_total[valid])
        else:
            mse_loss = torch.tensor(0.0, device=ou_logit.device)
        return self.bce_weight * bce_loss + self.mse_weight * mse_loss
