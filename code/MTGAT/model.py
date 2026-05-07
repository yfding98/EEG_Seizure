#!/usr/bin/env python3
"""MTAD-GAT encoder with seizure and SOZ heads."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, n_features: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve sequence length")
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(n_features, n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(self.padding(x)))
        return x.permute(0, 2, 1)


class FeatureAttentionLayer(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        dropout: float,
        alpha: float,
        embed_dim: int | None = None,
        use_gatv2: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias
        self.embed_dim = embed_dim if embed_dim is not None else window_size

        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.bias = nn.Parameter(torch.empty(n_features, n_features)) if use_bias else None
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        if self.use_gatv2:
            a_input = self._make_attention_input(x)
            a_input = self.leakyrelu(self.lin(a_input))
            e = torch.matmul(a_input, self.a).squeeze(3)
        else:
            wx = self.lin(x)
            a_input = self._make_attention_input(wx)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)
        if self.bias is not None:
            e = e + self.bias
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h = self.sigmoid(torch.matmul(attention, x))
        return h.permute(0, 2, 1)

    def _make_attention_input(self, v: torch.Tensor) -> torch.Tensor:
        k = self.num_nodes
        left = v.repeat_interleave(k, dim=1)
        right = v.repeat(1, k, 1)
        combined = torch.cat((left, right), dim=2)
        if self.use_gatv2:
            return combined.view(v.size(0), k, k, 2 * self.window_size)
        return combined.view(v.size(0), k, k, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        dropout: float,
        alpha: float,
        embed_dim: int | None = None,
        use_gatv2: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.num_nodes = window_size
        self.use_bias = use_bias
        self.embed_dim = embed_dim if embed_dim is not None else n_features

        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.bias = nn.Parameter(torch.empty(window_size, window_size)) if use_bias else None
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gatv2:
            a_input = self._make_attention_input(x)
            a_input = self.leakyrelu(self.lin(a_input))
            e = torch.matmul(a_input, self.a).squeeze(3)
        else:
            wx = self.lin(x)
            a_input = self._make_attention_input(wx)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)
        if self.bias is not None:
            e = e + self.bias
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        return self.sigmoid(torch.matmul(attention, x))

    def _make_attention_input(self, v: torch.Tensor) -> torch.Tensor:
        k = self.num_nodes
        left = v.repeat_interleave(k, dim=1)
        right = v.repeat(1, k, 1)
        combined = torch.cat((left, right), dim=2)
        if self.use_gatv2:
            return combined.view(v.size(0), k, k, 2 * self.n_features)
        return combined.view(v.size(0), k, k, 2 * self.embed_dim)


class GRULayer(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.gru = nn.GRU(
            in_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0 if n_layers == 1 else dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return h.transpose(0, 1).reshape(x.size(0), self.n_layers * self.hid_dim)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(max(1, n_layers)):
            layers.extend([nn.Linear(dim, hid_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hid_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MTGATSOZ(nn.Module):
    """MTAD-GAT style encoder for next-sample seizure and SOZ prediction."""

    def __init__(
        self,
        n_features: int = 22,
        window_size: int = 200,
        n_soz: int = 22,
        kernel_size: int = 5,
        feat_gat_embed_dim: int | None = None,
        time_gat_embed_dim: int | None = None,
        use_gatv2: bool = True,
        gru_n_layers: int = 1,
        gru_hid_dim: int = 64,
        head_n_layers: int = 1,
        head_hid_dim: int = 64,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features,
            window_size,
            dropout,
            alpha,
            embed_dim=feat_gat_embed_dim,
            use_gatv2=use_gatv2,
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features,
            window_size,
            dropout,
            alpha,
            embed_dim=time_gat_embed_dim,
            use_gatv2=use_gatv2,
        )
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        encoded_dim = gru_hid_dim * gru_n_layers
        self.seizure_head = MLPHead(encoded_dim, head_hid_dim, 1, head_n_layers, dropout)
        self.soz_head = MLPHead(encoded_dim, head_hid_dim, n_soz, head_n_layers, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)
        return self.gru(h_cat)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encode(x)
        return {
            "seizure_logit": self.seizure_head(h).squeeze(-1),
            "soz_logits": self.soz_head(h),
        }
