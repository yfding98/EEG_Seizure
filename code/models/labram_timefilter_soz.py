#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM-TimeFilter-SOZ: 基于预训练LaBraM + TimeFilter图过滤的EEG癫痫起始区检测模型

Architecture:
    Input: X [B, 22, 20, 100]  (22 TCP导联, 20 patches, 100 samples/patch)

    1. Patch Embedding
       - Flatten [B, 440, 100]  → Linear → [B, 440, D=128]
       - 2D Position Encoding (channel_idx + patch_idx)

    2. LaBraM Backbone (optional pre-trained)
       - 12 frozen Transformer layers  (底层, 通用EEG表征)
       - K  trainable Transformer layers(顶层, 任务适配)

    3. TimeFilter Core
       a) Multi-head projection distance (H=4) → k-NN graph (α=0.15)
       b) Three domain-specific filters:
          - Filter_Temporal:     保留同导联相邻补丁边 (时序连续性)
          - Filter_Spatial:      保留同时间解剖邻近导联边 (10-20球面距离 < 5cm)
          - Filter_Pathological: 学习HFO/病理模式相关边 (可选gamma能量先验)
       c) Noisy gated routing + Top-p=0.85 动态分配

    4. Graph Convolution: 2-layer GAT (heads=4)

    5. SOZ Localization Head
       - Temporal attention pooling  (学习发作起始补丁权重)
       - Channel max-pooling          (22导联聚合)
       - BipolarToMonopolarMapper     (22 TCP → 19 monopolar)
       - Sigmoid → SOZ probability

    Output: monopolar_probs [B, 19]

Loss:
    - Primary: Focal Loss (γ=2.0)
    - Auxiliary: Domain adversarial loss (GRL + discriminator)

Reference:
    - TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration (NeurIPS 2024)
    - LaBraM: Large Brain Model for EEG (ICLR 2024)
    - DeepSOZ: Deep Learning for SOZ Localization (Abou Jaoude et al., 2020)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# 常量  (与 data_preprocess/config.py 一致)
# =============================================================================

# 标准19通道 (10-20)
STANDARD_19 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]
STD19_IDX = {ch: i for i, ch in enumerate(STANDARD_19)}

# TCP 22通道双极导联 (顺序与 eeg_pipeline.py / config.py 对齐)
TCP_PAIRS: List[Tuple[str, str]] = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),       # 左颞链   0-3
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),       # 右颞链   4-7
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),       # 左副矢状 8-11
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),       # 右副矢状 12-15
    ('A1', 'T3'),  ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),       # 中央链   16-21
    ('C4', 'T4'),  ('T4', 'A2'),
]
TCP_NAMES = [f"{a}-{b}" for a, b in TCP_PAIRS]
N_TCP = 22
N_STD = 19

# 10-20 电极球面坐标 (单位球, 用于空间距离计算)
# 近似 (x, y, z): x=left/right, y=anterior/posterior, z=superior/inferior
_ELECTRODE_3D: Dict[str, Tuple[float, float, float]] = {
    'FP1': (-0.31, 0.95, 0.00), 'FP2': (0.31, 0.95, 0.00),
    'F7':  (-0.81, 0.59, 0.00), 'F3':  (-0.55, 0.67, 0.50),
    'FZ':  (0.00, 0.71, 0.71),  'F4':  (0.55, 0.67, 0.50),
    'F8':  (0.81, 0.59, 0.00),
    'T3':  (-1.00, 0.00, 0.00), 'C3':  (-0.57, 0.00, 0.82),
    'CZ':  (0.00, 0.00, 1.00),  'C4':  (0.57, 0.00, 0.82),
    'T4':  (1.00, 0.00, 0.00),
    'T5':  (-0.81, -0.59, 0.00), 'P3': (-0.55, -0.67, 0.50),
    'PZ':  (0.00, -0.71, 0.71),  'P4': (0.55, -0.67, 0.50),
    'T6':  (0.81, -0.59, 0.00),
    'O1':  (-0.31, -0.95, 0.00), 'O2': (0.31, -0.95, 0.00),
    'A1':  (-1.05, 0.00, -0.30), 'A2': (1.05, 0.00, -0.30),
}


# =============================================================================
# 配置
# =============================================================================

@dataclass
class ModelConfig:
    """LaBraM-TimeFilter-SOZ 模型配置"""

    # ---- 输入 ----
    n_channels: int = 22          # 脑电通道数 (仅支持22或19)
    n_patches: int = 10           # 每导联补丁数
    patch_len: int = 200          # LaBraM原始patch长度 (200采样点 @200Hz = 1s)
    n_nodes: int = 220            # n_channels * n_patches

    # ---- LaBraM Backbone (对齐 labram-base checkpoint) ----
    embed_dim: int = 200          # 对齐 LaBraM-base
    out_chans: int = 8            # TemporalConv 输出通道数
    labram_checkpoint: str = ''   # 预训练权重路径 (空=随机初始化)
    checkpoint_type: str = 'labram-base'  # 'labram-base' 或 'vqnsp'
    n_transformer_layers: int = 12  # 对齐 LaBraM-base (12层)
    n_frozen_layers: int = 10     # 冻结底层数
    n_heads_transformer: int = 10 # 对齐 LaBraM-base (10头)
    ff_mult: float = 4.0          # FFN扩展倍数
    transformer_dropout: float = 0.0  # LaBraM原始无基础dropout
    drop_path_rate: float = 0.0   # DropPath rate (线性递增)
    init_values: float = 0.1      # LayerScale init值 (labram-base用)
    use_qk_norm: bool = True      # QK normalization (labram-base用)

    # ---- TimeFilter ----
    tf_n_heads: int = 4           # 多头投影距离的头数 H
    tf_alpha: float = 0.15        # k-NN保留比例 α
    tf_n_filters: int = 3         # 过滤器数目 (S/T/ST)
    spatial_dist_thresh: float = 0.55  # 球面距离阈值 (≈5cm)
    top_p: float = 0.5            # Top-p 动态路由 (对齐原始TimeFilter)
    temporal_k: int = 3           # Temporal过滤器允许的最大补丁距离 ±k
    n_timefilter_blocks: int = 2  # TimeFilter block堆叠数

    # ---- GAT ----
    gat_layers: int = 2
    gat_heads: int = 4
    gat_dropout: float = 0.1

    # ---- Localization Head ----
    head_hidden: int = 64
    head_dropout: float = 0.3
    n_output: int = 22            # 输出通道数 (22=双极直接输出, 19=单极映射)
    output_mode: str = 'bipolar'  # 'bipolar' (22ch) or 'monopolar' (19ch)

    # ---- Domain Adversarial ----
    use_domain_adversarial: bool = True
    domain_hidden: int = 64
    grl_lambda: float = 0.1       # 梯度反转强度

    # ---- Loss ----
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    domain_loss_weight: float = 0.1
    moe_loss_weight: float = 0.01  # MoE辅助损失权重

    # ---- TCP pairs (允许外部覆盖) ----
    tcp_pairs: List[Tuple[str, str]] = field(default_factory=lambda: list(TCP_PAIRS))


# =============================================================================
# 1. TemporalConv Patch Embedding (原始LaBraM结构)
# =============================================================================

class TemporalConv(nn.Module):
    """
    LaBraM原始的时域卷积嵌入层

    输入: [B, N_electrodes, N_patches, patch_size]
    输出: [B, N_electrodes * N_patches, embed_dim]

    三层Conv2d提取局部时频特征, 比单层Linear更能捕获EEG波形模式。
    """

    def __init__(self, in_chans: int = 1, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, A, T]  (N=electrodes, A=patches, T=patch_size)
        Returns:
            [B, N*A, D]  where D = T_out * out_chans
        """
        B, N, A, T = x.shape
        x = x.reshape(B, N * A, T)            # [B, N*A, T]
        x = x.unsqueeze(1)                    # [B, 1, N*A, T]
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        # x: [B, C_out, N*A, T_out] → [B, N*A, T_out * C_out]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, N*A, T_out, C_out]
        x = x.reshape(B, N * A, -1)              # [B, N*A, D]
        return x


# =============================================================================
# 2. LaBraM Backbone (原始Transformer结构)
# =============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    """LaBraM原始MLP: fc1 → GELU → fc2 → Dropout (注意fc1后无dropout)"""

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    LaBraM原始Attention

    支持两种模式 (对应两个checkpoint):
    - labram-base: qkv_bias=False, qk_norm=LayerNorm  → gamma_1/gamma_2
    - vqnsp:       qkv_bias=True (q_bias+v_bias), qk_norm=None
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_norm=None, qk_scale=None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, attn_head_dim: int = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ])
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, N, d_h]

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)   # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    LaBraM原始Transformer Block

    Pre-LN + DropPath + 可选LayerScale(gamma_1/gamma_2)
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_norm=None, qk_scale=None,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, init_values: float = 0.0,
                 norm_layer=nn.LayerNorm, attn_head_dim: int = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class LaBraMBackbone(nn.Module):
    """
    LaBraM Backbone — 忠实于原始论文结构, 可加载预训练权重

    结构:
        TemporalConv → pos_embed + time_embed → N层 Block → LayerNorm

    支持两种checkpoint:
        - labram-base.pth (student.*前缀, LayerScale+QKNorm)
        - vqnsp.pth       (encoder.*前缀, QKVBias, 无LayerScale)

    底层 n_frozen 层冻结参数, 顶层可训练。
    不含 cls_token 和分类头 (用于下游密集预测)。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding
        self.patch_embed = TemporalConv(in_chans=1, out_chans=cfg.out_chans)

        # TemporalConv output dim depends on patch_len; project to embed_dim if needed
        _conv1_out = (cfg.patch_len + 2 * 7 - 15) // 8 + 1  # conv1 stride=8
        _temporal_out_dim = _conv1_out * cfg.out_chans
        if _temporal_out_dim != D:
            self.embed_proj = nn.Linear(_temporal_out_dim, D)
        else:
            self.embed_proj = nn.Identity()

        # 位置编码 (空间 + 时间)
        max_electrodes = max(129, cfg.n_channels + 1)  # 至少129 (对齐原始LaBraM)
        max_time_windows = max(16, cfg.n_patches)      # 至少16 (原始), 扩展至n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, max_electrodes, D))
        self.time_embed = nn.Parameter(torch.zeros(1, max_time_windows, D))
        self.pos_drop = nn.Dropout(p=cfg.transformer_dropout)

        # 根据checkpoint类型选择Block配置
        if cfg.checkpoint_type == 'labram-base':
            qkv_bias = False
            qk_norm = partial(nn.LayerNorm, eps=1e-6) if cfg.use_qk_norm else None
            init_values = cfg.init_values
        else:  # vqnsp
            qkv_bias = True
            qk_norm = None
            init_values = 0.0

        # DropPath 线性递增
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.n_transformer_layers)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=D, num_heads=cfg.n_heads_transformer,
                mlp_ratio=cfg.ff_mult, qkv_bias=qkv_bias, qk_norm=qk_norm,
                drop=cfg.transformer_dropout, attn_drop=0.0,
                drop_path=dpr[i], init_values=init_values,
                norm_layer=norm_layer,
            )
            for i in range(cfg.n_transformer_layers)
        ])
        self.norm = norm_layer(D)

        # 初始化
        self._init_weights()

        # 加载预训练权重
        if cfg.labram_checkpoint:
            self._load_checkpoint(cfg.labram_checkpoint, cfg.checkpoint_type)

        # 冻结底层
        self._freeze_bottom(cfg.n_frozen_layers)

    def _init_weights(self):
        """LaBraM原始初始化"""
        def _trunc_normal_(tensor, std=0.02):
            """截断正态分布初始化 (替代timm.trunc_normal_)"""
            nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2*std, b=2*std)

        if self.pos_embed is not None:
            _trunc_normal_(self.pos_embed, std=0.02)
        if self.time_embed is not None:
            _trunc_normal_(self.time_embed, std=0.02)
        self.apply(self.__init_module_weights)
        self._fix_init_weight()

    @staticmethod
    def __init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _fix_init_weight(self):
        """按深度rescale残差路径权重 (LaBraM原始策略)"""
        for layer_id, layer in enumerate(self.blocks):
            if hasattr(layer.attn, 'proj') and hasattr(layer.attn.proj, 'weight'):
                layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            if hasattr(layer.mlp, 'fc2') and hasattr(layer.mlp.fc2, 'weight'):
                layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _load_checkpoint(self, path: str, ckpt_type: str = 'labram-base'):
        """加载LaBraM预训练权重, 处理前缀映射"""
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
            if 'model' in state:
                state = state['model']
            elif 'state_dict' in state:
                state = state['state_dict']

            # 确定前缀
            prefix = 'student.' if ckpt_type == 'labram-base' else 'encoder.'

            # 映射: 去掉前缀, 过滤不需要的键 (cls_token, mask_token, lm_head等)
            skip_keys = {'cls_token', 'mask_token', 'lm_head', 'logit_scale',
                         'projection_head', 'head'}
            mapped = {}
            for k, v in state.items():
                if not k.startswith(prefix):
                    continue
                new_key = k[len(prefix):]
                # 跳过不需要的权重
                base = new_key.split('.')[0]
                if base in skip_keys:
                    continue
                mapped[new_key] = v

            # Interpolate pos_embed if shapes mismatch
            if 'pos_embed' in mapped:
                ckpt_pos = mapped['pos_embed']
                my_pos = self.pos_embed
                if ckpt_pos.shape != my_pos.shape:
                    logger.info(f"Interpolating pos_embed from {ckpt_pos.shape} to {my_pos.shape}")
                    # [1, N_elec, D] → interpolate along both axes if needed
                    if ckpt_pos.shape[2] != my_pos.shape[2]:
                        ckpt_pos = F.interpolate(
                            ckpt_pos.permute(0, 2, 1),
                            size=my_pos.shape[1],
                            mode='linear', align_corners=False,
                        ).permute(0, 2, 1)
                        # now ckpt_pos: [1, my_N, ckpt_D] → project D
                        ckpt_pos = F.interpolate(
                            ckpt_pos,  # [1, my_N, ckpt_D]
                            size=my_pos.shape[2],
                            mode='linear', align_corners=False,
                        )
                    elif ckpt_pos.shape[1] != my_pos.shape[1]:
                        ckpt_pos = F.interpolate(
                            ckpt_pos.permute(0, 2, 1),
                            size=my_pos.shape[1],
                            mode='linear', align_corners=False,
                        ).permute(0, 2, 1)
                    mapped['pos_embed'] = ckpt_pos

            # Interpolate time_embed if shapes mismatch
            if 'time_embed' in mapped:
                ckpt_time_embed = mapped['time_embed']
                my_time_embed = self.time_embed
                if ckpt_time_embed.shape != my_time_embed.shape:
                    logger.info(f"Interpolating time_embed from {ckpt_time_embed.shape} to {my_time_embed.shape}")
                    if ckpt_time_embed.shape[2] != my_time_embed.shape[2]:
                        ckpt_time_embed = F.interpolate(
                            ckpt_time_embed.permute(0, 2, 1),
                            size=my_time_embed.shape[1],
                            mode='linear', align_corners=False,
                        ).permute(0, 2, 1)
                        ckpt_time_embed = F.interpolate(
                            ckpt_time_embed,
                            size=my_time_embed.shape[2],
                            mode='linear', align_corners=False,
                        )
                    else:
                        ckpt_time_embed = F.interpolate(
                            ckpt_time_embed.permute(0, 2, 1),
                            size=my_time_embed.shape[1],
                            mode='linear', align_corners=False,
                        ).permute(0, 2, 1)
                    mapped['time_embed'] = ckpt_time_embed

            missing, unexpected = self.load_state_dict(mapped, strict=False)
            n_loaded = len(mapped) - len(unexpected)
            logger.info(
                f"LaBraM checkpoint loaded ({ckpt_type}): {path}\n"
                f"  loaded={n_loaded}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if missing:
                logger.debug(f"  Missing keys: {missing[:10]}...")
        except Exception as e:
            logger.warning(f"无法加载LaBraM checkpoint: {e}, 使用随机初始化")

    def _freeze_bottom(self, n_freeze: int):
        """冻结底部 n_freeze 层 + patch_embed + pos/time embed"""
        # 冻结嵌入层
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        self.pos_embed.requires_grad = False
        self.time_embed.requires_grad = False

        # 冻结底层blocks
        for i, layer in enumerate(self.blocks):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad = False

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LaBraM: {len(self.blocks)} layers, "
            f"frozen={n_freeze}+embed, trainable params={n_trainable:,}/{n_total:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        input_chans=None,
        use_time_embed: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N_channels, N_patches, patch_size]  e.g. [B, 22, 10, 200]
            input_chans: 可选, 电极通道索引 (用于子集电极)
        Returns:
            [B, N_channels * N_patches, embed_dim]
        """
        batch_size, n_ch, n_patches, patch_size = x.shape

        # TemporalConv → project to embed_dim
        x = self.patch_embed(x)                              # [B, n_ch*n_patches, T_out*C]
        x = self.embed_proj(x)                               # [B, n_ch*n_patches, D]

        # 位置编码
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            # 空间位置 (使用前n_ch个电极的编码, 不含CLS位)
            pos_e = pos_embed_used[:, 1:n_ch+1, :]           # [1, n_ch, D]
            pos_e = pos_e.unsqueeze(2).expand(batch_size, -1, n_patches, -1)
            pos_e = pos_e.flatten(1, 2)                       # [B, n_ch*n_patches, D]
            x = x + pos_e

        if self.time_embed is not None and use_time_embed:
            time_e = self.time_embed[:, :n_patches, :]        # [1, n_patches, D]
            time_e = time_e.unsqueeze(1).expand(batch_size, n_ch, -1, -1)
            time_e = time_e.flatten(1, 2)                     # [B, n_ch*n_patches, D]
            x = x + time_e

        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)


# =============================================================================
# 3. TimeFilter Core (恢复原始MoE机制 + 改进Temporal过滤)
# =============================================================================

# ---- 3a. mask_topk: k-NN 稀疏化 ----

def mask_topk(x: torch.Tensor, alpha: float = 0.5, largest: bool = False) -> torch.Tensor:
    """保留每行最大的 alpha 比例的值, 其余置零。 x: [B, H, L, L]"""
    k = max(1, int(alpha * x.shape[-1]))
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)
    return mask


# ---- 3b. 区域掩码生成 (S/T/ST) + 领域先验增强 ----

def build_region_masks(
    L: int, n_vars: int, device: torch.device,
    n_channels: int = 22, n_patches: int = 10,
    tcp_pairs: List[Tuple[str, str]] = None,
    spatial_dist_thresh: float = 0.55,
    temporal_k: int = 3,
) -> torch.Tensor:
    """
    构建3区域掩码 [L, 3, L], 融合原始TimeFilter的S/T/ST + EEG领域先验

    区域定义 (对节点k而言):
      S (Spatial):  同一时间片、不同导联的节点 (增强: 仅保留空间距离<阈值的导联对)
      T (Temporal): 同一导联、不同时间片的节点 (增强: 仅保留距离<=temporal_k, 带衰减)
      ST (Other):   其余所有节点
    """
    N = L // n_vars

    # 计算空间邻近矩阵
    pairs = tcp_pairs or list(TCP_PAIRS)
    ch_adj = np.ones((n_channels, n_channels), dtype=np.float32)
    if spatial_dist_thresh > 0:
        ch_pos = []
        for a, b in pairs[:n_channels]:
            pa = np.array(_ELECTRODE_3D.get(a, (0, 0, 0)))
            pb = np.array(_ELECTRODE_3D.get(b, (0, 0, 0)))
            ch_pos.append((pa + pb) / 2.0)
        ch_pos = np.array(ch_pos)
        from scipy.spatial.distance import cdist as scipy_cdist
        ch_dist = scipy_cdist(ch_pos, ch_pos)
        ch_adj = (ch_dist < spatial_dist_thresh).astype(np.float32)
        np.fill_diagonal(ch_adj, 0.0)

    masks = []
    for k_idx in range(L):
        ch_k = k_idx // N
        patch_k = k_idx % N

        S = torch.zeros(L, dtype=torch.float32, device=device)
        for ci in range(n_vars):
            if ci != ch_k and ch_adj[min(ci, n_channels-1), min(ch_k, n_channels-1)] > 0:
                S[ci * N + patch_k] = 1.0

        T = torch.zeros(L, dtype=torch.float32, device=device)
        for pi in range(N):
            if pi != patch_k:
                dist = abs(pi - patch_k)
                if dist <= temporal_k:
                    T[ch_k * N + pi] = math.exp(-dist / max(temporal_k, 1))

        ST = torch.ones(L, dtype=torch.float32, device=device)
        ST[k_idx] = 0.0
        ST = ST - S - T
        ST = ST.clamp(min=0.0)

        masks.append(torch.stack([S, T, ST], dim=0))

    return torch.stack(masks, dim=0)  # [L, 3, L]
# ---- 3c. MoE 门控路由 (原始TimeFilter机制 + 辅助损失) ----

class MoERouter(nn.Module):
    """
    原始TimeFilter的 mask_moe 机制

    - 对每个节点输出3个专家(S/T/ST)的门控权重
    - 含噪门控 + Top-p截断
    - 返回辅助损失 (cv_squared负载均衡 + cross_entropy多样性)
    """

    def __init__(self, n_vars: int, top_p: float = 0.5,
                 num_experts: int = 3, in_dim: int = 96):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.top_p = top_p

        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        """变异系数的平方 — 衡量负载均衡"""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    @staticmethod
    def cross_entropy_loss(x: torch.Tensor) -> torch.Tensor:
        """交叉熵 — 鼓励路由多样化"""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_p_gating(self, x: torch.Tensor, is_training: bool,
                           noise_epsilon: float = 1e-2):
        """含噪Top-p门控"""
        clean_logits = self.gate(x)
        if is_training:
            raw_noise = self.noise(x)
            noise_stddev = self.softplus(raw_noise) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy_loss(logits)

        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = F.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask, dtype=torch.float32)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[zero_indices[0], zero_indices[1],
                   sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1

        sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1
        loss = loss_importance + lambda_2 * loss_dynamic

        return top_p_mask, loss

    def forward(self, x: torch.Tensor, masks: torch.Tensor,
                is_training: bool = False):
        """
        Args:
            x: [B, H, L, L]  邻接矩阵
            masks: [L, 3, L]  区域掩码
        Returns:
            mask: [B, H, L, L]  路由后掩码
            loss: 辅助损失 (scalar)
        """
        B, H, L, _ = x.shape
        device = x.device

        mask_base = torch.eye(L, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if self.top_p == 0.0:
            return mask_base, torch.tensor(0.0, device=device)

        x_flat = x.reshape(B * H, L, L)
        gates, loss = self.noisy_top_p_gating(x_flat, is_training)
        gates = gates.reshape(B, H, L, -1).float()  # [B, H, L, 3]

        # 门控权重 × 区域掩码  →  最终掩码
        final_mask = torch.einsum('bhli,lid->bhld', gates, masks) + mask_base

        return final_mask, loss


# ---- 3d. Graph Learner (原始TimeFilter内积距离) ----

class GraphLearner(nn.Module):
    """原始TimeFilter的图学习器: 两个独立线性投影 → 内积 + GELU → k-NN稀疏化 → MoE路由"""

    def __init__(self, dim: int, n_vars: int, top_p: float = 0.5, in_dim: int = 96):
        super().__init__()
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.mask_moe = MoERouter(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x: torch.Tensor, masks: torch.Tensor,
                alpha: float = 0.5, is_training: bool = False):
        """x: [B, H, L, D] → adj: [B, H, L, L], loss"""
        adj = F.gelu(torch.einsum('bhid,bhjd->bhij', self.proj_1(x), self.proj_2(x)))
        adj = adj * mask_topk(adj, alpha)
        mask, loss = self.mask_moe(adj, masks, is_training)
        adj = adj * mask
        return adj, loss


# ---- 3e. GCN (原始TimeFilter图卷积) ----

class TimeFilterGCN(nn.Module):
    """原始TimeFilter的多头图卷积"""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """adj: [B, H, L, L], x: [B, L, D] → [B, L, D]"""
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()
        return x.view(B, L, -1)


# ---- 3f. GraphFilter (图学习 + 图卷积) ----

class GraphFilter(nn.Module):
    """图学习器 + 图卷积 组合模块"""

    def __init__(self, dim: int, n_vars: int, n_heads: int = 4,
                 top_p: float = 0.5, dropout: float = 0.0, in_dim: int = 96):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = GraphLearner(
            self.dim // self.n_heads, n_vars, top_p, in_dim=in_dim,
        )
        self.graph_conv = TimeFilterGCN(self.dim, self.n_heads)

    def forward(self, x: torch.Tensor, masks: torch.Tensor,
                alpha: float = 0.5, is_training: bool = False):
        """x: [B, L, D] → [B, L, D], loss"""
        B, L, D = x.shape
        x_h = x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3)  # [B, H, L, D//H]
        adj, loss = self.graph_learner(x_h, masks, alpha, is_training)
        adj = torch.softmax(adj, dim=-1)
        adj = self.dropout(adj)
        out = self.graph_conv(adj, x)
        return out, loss


# ---- 3g. GraphBlock (图过滤 + FFN + 残差) ----

class GraphBlock(nn.Module):
    """完整的TimeFilter GraphBlock: GraphFilter + LayerNorm + FFN + 残差"""

    def __init__(self, dim: int, n_vars: int, d_ff: int = None,
                 n_heads: int = 4, top_p: float = 0.5,
                 dropout: float = 0.0, in_dim: int = 96):
        super().__init__()
        self.gnn = GraphFilter(dim, n_vars, n_heads, top_p=top_p,
                               dropout=dropout, in_dim=in_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, d_ff or dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff or dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, masks: torch.Tensor,
                alpha: float = 0.5, is_training: bool = False):
        """x: [B, L, D] → [B, L, D], loss"""
        out, loss = self.gnn(self.norm1(x), masks, alpha, is_training)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x, loss


# ---- 3h. TimeFilterBackbone ----

class TimeFilterBackbone(nn.Module):
    """TimeFilter backbone: 多层 GraphBlock 堆叠, 返回MoE辅助损失"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.embed_dim
        n_vars = cfg.n_channels
        L = cfg.n_nodes

        self.blocks = nn.ModuleList([
            GraphBlock(
                dim=D, n_vars=n_vars, d_ff=D * 2,
                n_heads=cfg.tf_n_heads, top_p=cfg.top_p,
                dropout=cfg.gat_dropout, in_dim=L,
            )
            for _ in range(cfg.n_timefilter_blocks)
        ])
        self.n_blocks = cfg.n_timefilter_blocks
        self._masks: Optional[torch.Tensor] = None
        self._cfg = cfg

    def _get_masks(self, L: int, device: torch.device) -> torch.Tensor:
        if self._masks is None or self._masks.device != device:
            self._masks = build_region_masks(
                L=L, n_vars=self._cfg.n_channels, device=device,
                n_channels=self._cfg.n_channels, n_patches=self._cfg.n_patches,
                tcp_pairs=self._cfg.tcp_pairs,
                spatial_dist_thresh=self._cfg.spatial_dist_thresh,
                temporal_k=self._cfg.temporal_k,
            )
        return self._masks

    def forward(self, x: torch.Tensor, is_training: bool = False):
        """x: [B, L, D] → [B, L, D], moe_loss"""
        L = x.shape[1]
        masks = self._get_masks(L, x.device)
        moe_loss = 0.0
        for block in self.blocks:
            x, loss = block(x, masks, self._cfg.tf_alpha, is_training)
            moe_loss = moe_loss + loss
        moe_loss = moe_loss / max(self.n_blocks, 1)
        return x, moe_loss







# =============================================================================
# 4. Graph Attention Network (GAT)
# =============================================================================

class GATLayer(nn.Module):
    """
    Graph Attention Layer (masked by adjacency)

    attention(i,j) = LeakyReLU(a^T [Wh_i || Wh_j]) * adj(i,j)
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1,
                 concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        head_dim = out_dim // n_heads if concat else out_dim

        self.W = nn.Linear(in_dim, head_dim * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim if concat else out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in]
            adj: [B, N, N]
        Returns:
            [B, N, D_out]
        """
        B, N, _ = x.shape
        H = self.n_heads
        head_dim = self.W.out_features // H

        Wh = self.W(x).view(B, N, H, head_dim)  # [B, N, H, d_h]

        # 注意力分数
        e_src = (Wh * self.a_src).sum(dim=-1)    # [B, N, H]
        e_dst = (Wh * self.a_dst).sum(dim=-1)    # [B, N, H]
        attn = self.leaky_relu(
            e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # [B, N, N, H]
        )

        # 掩码: 非邻居置为 -inf
        mask = (adj > 0).unsqueeze(-1).expand_as(attn)  # [B, N, N, H]
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=2)            # softmax over j (neighbors)
        attn = torch.nan_to_num(attn, 0.0)       # 处理全 -inf 行
        attn = self.dropout(attn)

        # 聚合
        # attn: [B, N, N, H], Wh: [B, N, H, d_h]
        out = torch.einsum('bnjh,bjhd->bnhd', attn, Wh)  # [B, N, H, d_h]

        if self.concat:
            out = out.reshape(B, N, H * head_dim)
        else:
            out = out.mean(dim=2)

        return self.norm(out)


class GATBlock(nn.Module):
    """多层 GAT + 残差"""

    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            concat = (i < n_layers - 1)  # 最后一层不concat
            self.layers.append(GATLayer(dim, dim, n_heads, dropout, concat=concat))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = x
            x = layer(x, adj)
            x = F.elu(x)
            x = self.dropout(x) + residual  # 残差
        return x


# =============================================================================
# 5. SOZ Localization Head
# =============================================================================

class TemporalAttentionPooling(nn.Module):
    """
    时序注意力池化: 学习每个补丁时间点的重要性权重

    对每个导联的 P=20 个补丁, 学习注意力权重 → 加权聚合。
    关键: 允许模型自动关注发作起始时刻附近的补丁。
    """

    def __init__(self, dim: int, n_patches: int = 20):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, P, D]  (C=22 channels, P=20 patches)
        Returns:
            pooled: [B, C, D]
            attn_weights: [B, C, P]
        """
        scores = self.attn(x).squeeze(-1)    # [B, C, P]
        weights = F.softmax(scores, dim=-1)  # [B, C, P]
        pooled = (x * weights.unsqueeze(-1)).sum(dim=2)  # [B, C, D]
        return pooled, weights



# 导入极性感知的 BipolarToMonopolarMapper (详见 bipolar_to_monopolar.py)
try:
    from .bipolar_to_monopolar import BipolarToMonopolarMapper
except ImportError:
    from bipolar_to_monopolar import BipolarToMonopolarMapper


class SOZLocalizationHead(nn.Module):
    """
    SOZ定位头: 从440节点特征 → SOZ概率

    支持两种输出模式:
    - bipolar  (n_output=22): 直接输出22通道双极导联SOZ logits
    - monopolar (n_output=19): 22双极 → BipolarToMonopolar映射 → 19单极

    流程:
    1. Reshape: [B, 440, D] → [B, 22, 20, D]
    2. Temporal attention pooling: [B, 22, 20, D] → [B, 22, D]
    3. Channel max-pooling (辅助): [B, 22, 20, D] → [B, 22, D]
    4. 融合: concat → FC → [B, 22]
    5. (仅monopolar模式) BipolarToMonopolar: [B, 22] → [B, 19]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.embed_dim
        self.n_channels = cfg.n_channels
        self.n_patches = cfg.n_patches
        self.output_mode = cfg.output_mode  # 'bipolar' or 'monopolar'
        self.n_output = cfg.n_output

        self.temporal_attn = TemporalAttentionPooling(D, cfg.n_patches)

        # 融合两种池化
        self.fusion = nn.Sequential(
            nn.Linear(D * 2, cfg.head_hidden),
            nn.LayerNorm(cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden // 2),
            nn.LayerNorm(cfg.head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden // 2, 1),
        )

        # 仅在 monopolar 模式下使用 BipolarToMonopolar 映射
        self.bipolar_to_mono = None
        if self.output_mode == 'monopolar':
            self.bipolar_to_mono = BipolarToMonopolarMapper(
                monopolar_channels=list(STANDARD_19),
                bipolar_pairs=cfg.tcp_pairs,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, N=440, D]
        Returns:
            soz_logits: [B, n_output]  (22 bipolar or 19 monopolar)
            aux: dict with bipolar_logits, attention weights
        """
        B = x.size(0)
        D = x.size(-1)

        # Reshape to channel x patch
        x_4d = x.view(B, self.n_channels, self.n_patches, D)  # [B, 22, 20, D]

        # 时序注意力池化
        attn_pooled, attn_weights = self.temporal_attn(x_4d)   # [B, 22, D], [B, 22, 20]

        # 通道最大池化
        max_pooled = x_4d.max(dim=2).values                     # [B, 22, D]

        # 融合 → 22通道双极logits
        fused = torch.cat([attn_pooled, max_pooled], dim=-1)    # [B, 22, 2D]
        bipolar_logits = self.fusion(fused).squeeze(-1)          # [B, 22]

        aux = {
            'bipolar_logits': bipolar_logits,
            'temporal_attn_weights': attn_weights,
        }

        if self.output_mode == 'bipolar':
            # 直接输出22通道双极logits
            return bipolar_logits, aux
        else:
            # 映射到19通道单极logits
            monopolar_logits = self.bipolar_to_mono.forward_logits(bipolar_logits)  # [B, 19]
            return monopolar_logits, aux


# =============================================================================
# 6. Domain Adversarial (GRL + Discriminator)
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层: 前向不变, 反向乘以 -λ"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """
    域判别器: 判断样本来自公共(TUSZ)还是私有数据

    输入: 全局池化后的特征 [B, D]
    输出: domain logit [B, 1]  (0=public, 1=private)
    """

    def __init__(self, dim: int, hidden: int = 64, grl_lambda: float = 0.1):
        super().__init__()
        self.grl = GradientReversalLayer(grl_lambda)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D]  全局特征
        Returns:
            [B, 1]  域预测 logit
        """
        x = self.grl(x)
        return self.net(x)


# =============================================================================
# 7. Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t (1-p_t)^γ log(p_t)

    适用于多标签分类 (每个通道独立二分类)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs:  [B, C]  logits
            targets: [B, C]  binary labels (0/1)
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        # alpha_t: alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * bce

        if sample_weight is not None:
            loss = loss * sample_weight.unsqueeze(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SOZDetectionLoss(nn.Module):
    """
    SOZ检测组合损失

    L_total = L_focal(SOZ定位) + λ_domain * L_bce(域判别)

    Args:
        focal_gamma: Focal Loss γ参数
        focal_alpha: Focal Loss α参数
        domain_weight: 域对抗损失权重
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        domain_weight: float = 0.1,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.domain_bce = nn.BCEWithLogitsLoss()
        self.domain_weight = domain_weight

    def forward(
        self,
        soz_logits: torch.Tensor,
        soz_targets: torch.Tensor,
        domain_logits: Optional[torch.Tensor] = None,
        domain_targets: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            total_loss, {'focal': ..., 'domain': ..., 'total': ...}
        """
        loss_focal = self.focal(soz_logits, soz_targets, sample_weight=sample_weight)
        losses = {'focal': loss_focal}

        total = loss_focal

        if domain_logits is not None and domain_targets is not None:
            loss_domain = self.domain_bce(domain_logits, domain_targets)
            total = total + self.domain_weight * loss_domain
            losses['domain'] = loss_domain

        losses['total'] = total
        return total, losses


# =============================================================================
# 8. Main Model: LaBraM_TimeFilter_SOZ
# =============================================================================

class LaBraM_TimeFilter_SOZ(nn.Module):
    """
    LaBraM-TimeFilter-SOZ: 预训练大模型 + 图过滤的EEG癫痫起始区检测

    支持两种输出模式:
    - bipolar  (n_output=22): 直接输出22通道TCP双极导联SOZ概率
    - monopolar (n_output=19): 通过BipolarToMonopolar映射输出19通道单极SOZ概率

    Forward:
        Input:  X [B, 22, 20, 100]   TCP双极导联 × 补丁 × 采样点
                domain_labels [B, 1]  可选, 0=public 1=private
                gamma_energy [B, 440, 1]  可选, 预提取γ频段能量

        Output: {
            'soz_probs':      [B, n_output]  SOZ概率 (sigmoid)
            'soz_logits':     [B, n_output]  SOZ logits
            'bipolar_logits': [B, 22]        22 TCP通道 logits (中间层)
            'domain_logits':  [B, 1]         域判别 logits (如果启用)
            'temporal_attn':  [B, 22, 20]    时序注意力权重
            'filtered_adj':   [B, 440, 440]  过滤后邻接矩阵
        }

    Usage:
        # 22通道双极模式（默认，配合 combined_manifest.csv 的双极标签）
        cfg = ModelConfig(n_output=22, output_mode='bipolar')
        model = LaBraM_TimeFilter_SOZ(cfg)
        out = model(torch.randn(4, 22, 20, 100))
        probs = out['soz_probs']   # [4, 22]

        # 19通道单极模式（配合 eeg_pipeline.py 的单极标签）
        cfg = ModelConfig(n_output=19, output_mode='monopolar')
        model = LaBraM_TimeFilter_SOZ(cfg)
        out = model(torch.randn(4, 22, 20, 100))
        probs = out['soz_probs']   # [4, 19]
    """

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg

        # ---- 1. LaBraM Backbone (含TemporalConv嵌入层) ----
        self.backbone = LaBraMBackbone(cfg)

        # ---- 2. TimeFilter (GraphBlock堆叠 + MoE路由) ----
        self.timefilter = TimeFilterBackbone(cfg)

        # ---- 3. SOZ Localization Head ----
        self.soz_head = SOZLocalizationHead(cfg)

        # ---- 4. Domain Discriminator (optional) ----
        self.domain_disc = None
        if cfg.use_domain_adversarial:
            self.domain_disc = DomainDiscriminator(
                dim=cfg.embed_dim,
                hidden=cfg.domain_hidden,
                grl_lambda=cfg.grl_lambda,
            )

        self._init_weights()

    def _init_weights(self):
        """Xavier初始化 (仅可训练参数)"""
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(
        self,
        x: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        gamma_energy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 22, 20, 100]  TCP双极导联数据
            domain_labels: [B, 1]  0=public, 1=private (训练域判别器时需要)
            gamma_energy: [B, 440, 1]  可选, 预提取γ频段能量

        Returns:
            Dict with keys:
                soz_probs, soz_logits, bipolar_logits,
                domain_logits, temporal_attn, filtered_adj
        """
        B = x.size(0)

        # (1) LaBraM Backbone (含TemporalConv + 位置编码 + Transformer)
        h = self.backbone(x)                          # [B, 440, D]

        # (2) TimeFilter (GraphBlock堆叠, 原始MoE路由)
        h, moe_loss = self.timefilter(h, is_training=self.training)  # [B, 440, D], scalar

        # (3) SOZ Localization Head
        soz_logits, aux = self.soz_head(h)            # [B, n_output]
        soz_probs = torch.sigmoid(soz_logits)

        outputs = {
            'soz_probs': soz_probs,                   # [B, n_output]
            'soz_logits': soz_logits,                  # [B, n_output]
            'bipolar_logits': aux['bipolar_logits'],    # [B, 22] 始终可用
            'temporal_attn': aux['temporal_attn_weights'],
            'moe_loss': moe_loss,                      # MoE辅助损失
            # 向后兼容旧键名
            'monopolar_probs': soz_probs,
            'monopolar_logits': soz_logits,
        }

        # (4) Domain Discriminator
        if self.domain_disc is not None:
            global_feat = h.mean(dim=1)               # [B, D]
            domain_logits = self.domain_disc(global_feat)  # [B, 1]
            outputs['domain_logits'] = domain_logits

        return outputs

    def get_loss_fn(self) -> SOZDetectionLoss:
        """获取配套损失函数"""
        return SOZDetectionLoss(
            focal_gamma=self.cfg.focal_gamma,
            focal_alpha=self.cfg.focal_alpha,
            domain_weight=self.cfg.domain_loss_weight,
        )

    def set_grl_lambda(self, lambda_: float):
        """动态调整梯度反转强度 (通常随训练进度递增)"""
        if self.domain_disc is not None:
            self.domain_disc.grl.set_lambda(lambda_)

    def get_trainable_params(self) -> List[Dict]:
        """获取分组参数 (便于差异学习率)"""
        backbone_params = []
        head_params = []
        domain_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(p)
            elif 'domain_disc' in name:
                domain_params.append(p)
            else:
                head_params.append(p)

        groups = [
            {'params': backbone_params, 'lr_scale': 0.1, 'name': 'backbone'},
            {'params': head_params, 'lr_scale': 1.0, 'name': 'head'},
        ]
        if domain_params:
            groups.append({'params': domain_params, 'lr_scale': 1.0, 'name': 'domain'})

        return groups

    def summary(self) -> str:
        """模型摘要"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        lines = [
            "=" * 60,
            "LaBraM-TimeFilter-SOZ Model Summary",
            "=" * 60,
            f"Input:  [B, {self.cfg.n_channels}, {self.cfg.n_patches}, {self.cfg.patch_len}]",
            f"Output: soz_probs [B, {self.cfg.n_output}] ({self.cfg.output_mode} mode)",
            f"",
            f"Backbone: {self.cfg.n_transformer_layers} layers "
            f"({self.cfg.n_frozen_layers} frozen + "
            f"{self.cfg.n_transformer_layers - self.cfg.n_frozen_layers} trainable)",
            f"Embed dim: {self.cfg.embed_dim}",
            f"TimeFilter: H={self.cfg.tf_n_heads}, alpha={self.cfg.tf_alpha}, "
            f"F={self.cfg.tf_n_filters}, Top-p={self.cfg.top_p}",
            f"GAT: {self.cfg.gat_layers} layers, {self.cfg.gat_heads} heads",
            f"Domain adversarial: {self.cfg.use_domain_adversarial}",
            f"",
            f"Parameters: {total:,} total, {trainable:,} trainable, {frozen:,} frozen",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# 便捷构造函数
# =============================================================================

def build_model(
    checkpoint: str = '',
    checkpoint_type: str = 'labram-base',
    n_frozen: int = 10,
    embed_dim: int = 200,
    use_domain_adversarial: bool = True,
    **kwargs,
) -> LaBraM_TimeFilter_SOZ:
    """快速构建模型"""
    cfg = ModelConfig(
        labram_checkpoint=checkpoint,
        checkpoint_type=checkpoint_type,
        n_frozen_layers=n_frozen,
        embed_dim=embed_dim,
        use_domain_adversarial=use_domain_adversarial,
        **kwargs,
    )
    model = LaBraM_TimeFilter_SOZ(cfg)
    logger.info(model.summary())
    return model


# =============================================================================
# 自测
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    for mode, n_out in [('bipolar', 22), ('monopolar', 19)]:
        print(f"\n{'='*60}")
        print(f"Testing {mode} mode (n_output={n_out})")
        print(f"{'='*60}")

        cfg = ModelConfig(
            labram_checkpoint='',
            n_transformer_layers=4,
            n_frozen_layers=2,
            n_output=n_out,
            output_mode=mode,
            patch_len=200,
            n_timefilter_blocks=1,
        )
        model = LaBraM_TimeFilter_SOZ(cfg)
        print(model.summary())

        B = 4
        X = torch.randn(B, 22, 20, 200)  # patch_len=200 对齐LaBraM
        domain = torch.tensor([[0], [0], [1], [1]], dtype=torch.float32)

        # Forward (shape check)
        print(f"\nForward pass...")
        with torch.no_grad():
            out = model(X, domain_labels=domain)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}" if v.dim() > 0 else f"  {k}: {v.item():.4f}")

        assert out['soz_probs'].shape == (B, n_out), \
            f"Expected soz_probs shape ({B}, {n_out}), got {out['soz_probs'].shape}"
        print(f"  MoE aux loss: {out['moe_loss']:.6f}")

        # Loss + Backward
        print(f"\nLoss + Backward...")
        out = model(X, domain_labels=domain)
        loss_fn = model.get_loss_fn()
        y_soz = torch.zeros(B, n_out)
        y_soz[0, [0, 2]] = 1.0
        y_soz[2, [5, min(8, n_out - 1)]] = 1.0

        total_loss, loss_dict = loss_fn(
            out['soz_logits'], y_soz,
            out.get('domain_logits'), domain,
        )
        # 加上MoE辅助损失
        total_loss = total_loss + cfg.moe_loss_weight * out['moe_loss']
        print(f"  Total loss (incl. MoE): {total_loss.item():.4f}")
        total_loss.backward()
        print(f"  Backward OK")

    print(f"\n[OK] All tests passed (bipolar + monopolar modes)!")
