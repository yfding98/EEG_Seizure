#!/usr/bin/env python3
"""Simple 1D-CNN baselines for EEG seizure detection and SOZ localization."""

import torch
import torch.nn as nn

from baseline.regions import (
    BIPOLAR_REGION_INDICES,
    MONOPOLAR_REGION_INDICES_17,
    REGION_NAMES,
)


class SimpleCNN(nn.Module):
    """
    Minimal 1D-CNN baseline.

    Input:  (B, 22, T)  - 22 TCP bipolar channels, T time samples
    Output: detection mode  -> (B, 1)   sigmoid logits
            soz mode        -> (B, 22)  per-channel sigmoid logits
    """

    def __init__(self, n_channels=22, task='detection'):
        super().__init__()
        assert task in ('detection', 'soz')
        self.task = task

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        out_dim = 1 if task == 'detection' else n_channels
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        # x: (B, 22, T)
        feat = self.features(x).squeeze(-1)   # (B, 256)
        return self.head(feat)                 # (B, 1) or (B, 22)


class RegionTemporalBlock(nn.Module):
    """Shared temporal CNN used independently for each variable-size region."""

    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(32, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RegionCNN(nn.Module):
    """
    Variable-channel CNN for brain-region SOZ localization.

    Input:  (B, 22, T) - 22 TCP bipolar channels
    Output: (B, 5)     - logits for REGION_NAMES
    """

    def __init__(self, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.region_names = list(REGION_NAMES)
        self.region_indices = {
            name: list(BIPOLAR_REGION_INDICES[name])
            for name in self.region_names
        }

        self.region_cnns = nn.ModuleDict({
            name: RegionTemporalBlock(len(indices), hidden_dim)
            for name, indices in self.region_indices.items()
        })
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for name in self.region_names
        })

    def forward(self, x):
        logits = []
        for name in self.region_names:
            idx = self.region_indices[name]
            region_x = x[:, idx, :]
            feat = self.region_cnns[name](region_x)
            logits.append(self.heads[name](feat))
        return torch.cat(logits, dim=1)


class SeparableRegionTemporalBlock(nn.Module):
    """Depthwise-separable temporal CNN for a variable-size region."""

    def __init__(self, in_channels, hidden_dim=64, depth_multiplier=4):
        super().__init__()
        depthwise_channels = in_channels * depth_multiplier
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                depthwise_channels,
                kernel_size=15,
                stride=2,
                padding=7,
                groups=in_channels,
            ),
            nn.BatchNorm1d(depthwise_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(depthwise_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=7,
                stride=2,
                padding=3,
                groups=hidden_dim,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MonopolarRegionCNN(nn.Module):
    """
    Standard per-region Conv1d model over 17 TUSZ-available monopolar channels.

    This is the direct monopolar version of RegionCNN: each brain region has its
    own Conv1d stack whose input channel count follows that region.
    """

    def __init__(self, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.region_names = list(REGION_NAMES)
        self.region_indices = {
            name: list(MONOPOLAR_REGION_INDICES_17[name])
            for name in self.region_names
        }
        self.region_cnns = nn.ModuleDict({
            name: RegionTemporalBlock(len(indices), hidden_dim)
            for name, indices in self.region_indices.items()
        })
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for name in self.region_names
        })

    def forward(self, x):
        logits = []
        for name in self.region_names:
            region_x = x[:, self.region_indices[name], :]
            feat = self.region_cnns[name](region_x)
            logits.append(self.heads[name](feat))
        return torch.cat(logits, dim=1)


class MonopolarSeparableRegionCNN(nn.Module):
    """
    Depthwise-separable per-region model.

    Depthwise Conv1d learns temporal filters independently per electrode, then
    pointwise Conv1d mixes electrodes/features inside the region.
    """

    def __init__(self, hidden_dim=64, dropout=0.5, depth_multiplier=4):
        super().__init__()
        self.region_names = list(REGION_NAMES)
        self.region_indices = {
            name: list(MONOPOLAR_REGION_INDICES_17[name])
            for name in self.region_names
        }
        self.region_cnns = nn.ModuleDict({
            name: SeparableRegionTemporalBlock(
                len(indices),
                hidden_dim=hidden_dim,
                depth_multiplier=depth_multiplier,
            )
            for name, indices in self.region_indices.items()
        })
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for name in self.region_names
        })

    def forward(self, x):
        logits = []
        for name in self.region_names:
            region_x = x[:, self.region_indices[name], :]
            feat = self.region_cnns[name](region_x)
            logits.append(self.heads[name](feat))
        return torch.cat(logits, dim=1)


class ChannelTemporalEncoder(nn.Module):
    """Shared single-electrode temporal encoder used before region pooling."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(32, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        # x: (B*C, 1, T)
        return self.net(x).squeeze(-1)


class MonopolarSharedAttentionRegionCNN(nn.Module):
    """
    Shared channel encoder + attention region pooling.

    Every monopolar electrode is encoded by the same temporal CNN. Each region
    then pools a variable number of electrode embeddings with learned attention,
    which is the most natural option when region channel counts differ.
    """

    def __init__(self, n_channels=17, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.region_names = list(REGION_NAMES)
        self.region_indices = {
            name: list(MONOPOLAR_REGION_INDICES_17[name])
            for name in self.region_names
        }
        self.encoder = ChannelTemporalEncoder(hidden_dim=hidden_dim)
        self.channel_embedding = nn.Parameter(torch.zeros(n_channels, hidden_dim))
        self.attention = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for name in self.region_names
        })
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for name in self.region_names
        })

    def forward(self, x):
        bsz, n_ch, n_time = x.shape
        if n_ch != self.n_channels:
            raise ValueError(f'Expected {self.n_channels} channels, got {n_ch}')

        flat = x.reshape(bsz * n_ch, 1, n_time)
        enc = self.encoder(flat).reshape(bsz, n_ch, -1)
        enc = enc + self.channel_embedding.unsqueeze(0)

        logits = []
        for name in self.region_names:
            region_feat = enc[:, self.region_indices[name], :]
            score = self.attention[name](region_feat).squeeze(-1)
            weight = torch.softmax(score, dim=1).unsqueeze(-1)
            pooled = (region_feat * weight).sum(dim=1)
            logits.append(self.heads[name](pooled))
        return torch.cat(logits, dim=1)


class EEGNetRegionBlock(nn.Module):
    """EEGNet-style temporal + depthwise spatial block for one region."""

    def __init__(self, n_region_channels, f1=8, depth_multiplier=2, f2=32, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(n_region_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        # x: (B, C_region, T)
        return self.net(x.unsqueeze(1)).flatten(1)


class MonopolarEEGNetRegionCNN(nn.Module):
    """
    EEGNet-style region model over 17 monopolar channels.

    Included as an additional implementation option because it is often the
    best EEG-specific factorization: temporal filters first, then depthwise
    spatial filters across the variable-size region.
    """

    def __init__(self, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.region_names = list(REGION_NAMES)
        self.region_indices = {
            name: list(MONOPOLAR_REGION_INDICES_17[name])
            for name in self.region_names
        }
        self.region_blocks = nn.ModuleDict({
            name: EEGNetRegionBlock(len(indices), f2=hidden_dim, dropout=dropout)
            for name, indices in self.region_indices.items()
        })
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 1)
            for name in self.region_names
        })

    def forward(self, x):
        logits = []
        for name in self.region_names:
            region_x = x[:, self.region_indices[name], :]
            feat = self.region_blocks[name](region_x)
            logits.append(self.heads[name](feat))
        return torch.cat(logits, dim=1)
