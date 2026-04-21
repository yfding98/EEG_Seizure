#!/usr/bin/env python3
"""Simple 1D-CNN baseline for EEG seizure detection and SOZ localization."""

import torch
import torch.nn as nn


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
