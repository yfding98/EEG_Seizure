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
            time_e = time_e.unsqueeze(1).exp