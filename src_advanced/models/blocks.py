"""
Reusable building blocks for advanced model architectures.

- ResidualBlock: skip-connection MLP block
- SEBlock: Squeeze-and-Excitation for feature attention
- ResidualSEBlock: Residual + SE combined
- DeepHead: stacked ResidualSE blocks + output projection
- FocalLoss: class-balanced focal loss for imbalanced classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """BN -> Linear -> GELU -> Dropout -> BN -> Linear -> Dropout -> (+skip)."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.drop1(self.act(self.linear1(self.bn1(x))))
        out = self.drop2(self.linear2(self.bn2(out)))
        return out + residual


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for feature attention."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.fc1 = nn.Linear(dim, mid)
        self.fc2 = nn.Linear(mid, dim)

    def forward(self, x):
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return x * scale


class ResidualSEBlock(nn.Module):
    """ResidualBlock + SEBlock combined."""

    def __init__(self, dim: int, dropout: float = 0.2, se_reduction: int = 4):
        super().__init__()
        self.residual = ResidualBlock(dim, dropout)
        self.se = SEBlock(dim, se_reduction)

    def forward(self, x):
        return self.se(self.residual(x))


class StackedResidualSE(nn.Module):
    """
    Progressive encoder: input_dim -> [512, 256, 128] with ResidualSE blocks
    at each layer and projection between layers.
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int],
        dropout: float = 0.2,
        se_reduction: int = 4,
        n_blocks_per_layer: int = 1,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        in_dim = input_dim
        for dim in layer_dims:
            # Projection + GELU + Dropout
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            # ResidualSE blocks at this dimension
            for _ in range(n_blocks_per_layer):
                layers.append(ResidualSEBlock(dim, dropout, se_reduction))
            in_dim = dim

        self.net = nn.Sequential(*layers)
        self._output_dim = layer_dims[-1]

    def forward(self, x):
        return self.net(self.input_bn(x))


class DeepHead(nn.Module):
    """
    Linear(in -> hid) -> [ResidualBlock] x N -> Linear(hid -> out).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(hidden_dim, dropout))
        blocks.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Combines with class weights for further imbalance correction.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
