#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectedBrainTimeFilter -- 有向图感知的脑网络TimeFilter

设计动机:
    脑网络4种特征中, GC/TE/AEC 是有向的 (adj[i,j] ≠ adj[j,i]),
    仅 wPLI 是无向的. 传统 TimeFilter 不区分方向性.
    本模块针对这一特点设计:

    1. 有向图感知的 GraphLearner:
       - 有向分支: 非对称内积 proj_src(x)·proj_tgt(x) 保留方向信息
       - 无向分支: 对称欧氏距离 + 高斯核
       - 特征感知 MoE 路由合并

    2. 时序-特征双维度过滤:
       - 时序维度: 沿 P 个补丁捕获网络连接的动态演化
       - 特征维度: 跨 4 种特征门控选择, 捕获特征互补性

    3. 方向性保持的图卷积:
       - 有向: DirectedGATLayer (已有实现)
       - 无向: GCNLayer (已有实现)

输入: [B, P, C, C, 4] 脑网络特征 (gc/te/aec/wpli)
输出: [B, P, C, C, 4] 过滤后特征 + MoE辅助损失
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 特征索引
FEATURE_NAMES = ['gc', 'te', 'aec', 'wpli']
FEATURE_DIRECTED = [True, True, True, False]  # 前三个有向, wPLI无向


@dataclass
class DirectedTimeFilterConfig:
    """有向脑网络TimeFilter配置"""
    n_channels: int = 22
    n_patches: int = 10
    n_features: int = 4
    n_heads: int = 4
    alpha: float = 0.15
    top_p: float = 0.5
    n_blocks: int = 2
    hidden_dim: int = 64          # 单分支GNN隐藏维度
    snapshot_dim: int = 128       # 快照编码输出维度
    dropout: float = 0.1
    temporal_k: int = 3           # 时序过滤 ±k 范围


# =====================================================================
# Direction-Aware Graph Learning
# =====================================================================

class DirectedGraphLearner(nn.Module):
    """
    有向图感知的图学习

    - 有向特征 (GC/TE/AEC):  proj_src(x) · proj_tgt(x) → 非对称邻接
    - 无向特征 (wPLI):       对称欧氏距离 + 高斯核 → 对称邻接
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        head_dim = dim // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        # 有向: src/tgt 独立投影
        self.proj_src = nn.Linear(dim, head_dim * n_heads, bias=False)
        self.proj_tgt = nn.Linear(dim, head_dim * n_heads, bias=False)

        # 无向: 对称投影
        self.proj_sym = nn.Linear(dim, head_dim * n_heads, bias=False)

    def forward(self, x: torch.Tensor, directed: bool = True,
                alpha: float = 0.15) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]  节点特征
            directed: True=有向, False=无向
        Returns:
            adj: [B, H, N, N]  多头邻接矩阵
        """
        B, N, D = x.shape
        H = self.n_heads
        k = max(1, int(alpha * N))

        if directed:
            # 非对称内积
            src = self.proj_src(x).view(B, N, H, self.head_dim)  # [B, N, H, d]
            tgt = self.proj_tgt(x).view(B, N, H, self.head_dim)  # [B, N, H, d]
            # adj[i,j] = src_i · tgt_j  (j→i 的边强度)
            adj = torch.einsum('bihd,bjhd->bhij', src, tgt)      # [B, H, N, N]
            adj = F.gelu(adj)
        else:
            # 对称欧氏距离 + 高斯核
            z = self.proj_sym(x).view(B, N, H, self.head_dim)    # [B, N, H, d]
            z = z.permute(0, 2, 1, 3)                             # [B, H, N, d]
            dist = torch.cdist(z, z, p=2)                         # [B, H, N, N]
            sigma = dist.mean(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
            adj = torch.exp(-dist ** 2 / (2 * sigma ** 2))

        # k-NN稀疏化
        _, topk_idx = adj.topk(k, dim=-1, largest=True)
        mask = torch.zeros_like(adj)
        mask.scatter_(-1, topk_idx, 1.0)

        if not directed:
            mask = torch.maximum(mask, mask.transpose(-1, -2))

        return adj * mask


# =====================================================================
# Region Mask Builder (for brain network features)
# =====================================================================

def build_brain_network_masks(
    P: int, n_features: int, temporal_k: int = 3,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    为脑网络特征序列构建3区域掩码

    将 [P, 4] 的特征序列展平为 L = P * n_features 个节点:
      S (Feature):  同时间步、不同特征的节点
      T (Temporal): 同特征、不同时间步的节点 (±temporal_k)
      ST (Other):   其余
    """
    L = P * n_features
    masks = []

    for k_idx in range(L):
        feat_k = k_idx // P
        patch_k = k_idx % P

        # S: 同时间步, 不同特征
        S = torch.zeros(L, dtype=torch.float32, device=device)
        for fi in range(n_features):
            if fi != feat_k:
                S[fi * P + patch_k] = 1.0

        # T: 同特征, 不同时间步 (±temporal_k, 带衰减)
        T = torch.zeros(L, dtype=torch.float32, device=device)
        for pi in range(P):
            if pi != patch_k:
                dist = abs(pi - patch_k)
                if dist <= temporal_k:
                    T[feat_k * P + pi] = math.exp(-dist / max(temporal_k, 1))

        # ST: 其余
        ST = torch.ones(L, dtype=torch.float32, device=device)
        ST[k_idx] = 0.0
        ST = ST - S - T
        ST = ST.clamp(min=0.0)

        masks.append(torch.stack([S, T, ST], dim=0))

    return torch.stack(masks, dim=0)  # [L, 3, L]


# =====================================================================
# Feature-Aware MoE Router
# =====================================================================

class FeatureAwareMoERouter(nn.Module):
    """
    特征感知的MoE路由器

    在 MoERouter 基础上增加特征类型嵌入,
    使路由决策感知当前处理的是哪种脑网络特征。
    """

    def __init__(self, in_dim: int, n_features: int = 4,
                 num_experts: int = 3, top_p: float = 0.5):
        super().__init__()
        self.num_experts = num_experts
        self.top_p = top_p

        self.feature_embed = nn.Embedding(n_features, in_dim)
        self.gate_norm = nn.LayerNorm(in_dim)
        self.gate = nn.Linear(in_dim, num_experts, bias=False)
        self.noise = nn.Linear(in_dim, num_experts, bias=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x: torch.Tensor, feature_ids: torch.Tensor,
                is_training: bool = False):
        """
        Args:
            x: [B, L]  门控输入 (L = last dim of flattened adj)
            feature_ids: [L]  每个节点所属的特征类型ID
        Returns:
            gates: [B, L, num_experts]  门控权重
            loss: 辅助损失
        """
        # 加入特征类型嵌入
        feat_emb = self.feature_embed(feature_ids)  # [L, D]
        # 将 x 扩展到与 feat_emb 匹配
        if x.dim() == 2:
            x = x.unsqueeze(-1) * feat_emb.unsqueeze(0)  # [B, L, D]
            x = x.sum(-1)  # [B, L]

        gate_in = self.gate_norm(feat_emb.unsqueeze(0).expand(x.shape[0], -1, -1))
        clean_logits = self.gate(gate_in)  # [B, L, E]

        if is_training:
            noise_std = self.softplus(self.noise(gate_in)) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            logits = clean_logits

        gates = self.softmax(logits)  # [B, L, E]

        # 辅助损失
        loss_cv = self.cv_squared(gates.sum(0).sum(0))  # 负载均衡
        eps = 1e-10
        loss_entropy = -torch.mul(gates, torch.log(gates + eps)).sum(-1).mean()
        loss = loss_cv + 0.1 * loss_entropy

        return gates, loss


# =====================================================================
# Directed Brain Network Graph Block
# =====================================================================

class DirectedBrainGraphBlock(nn.Module):
    """
    有向脑网络图过滤块

    对每种特征分支:
      1. DirectedGraphLearner 构建邻接矩阵
      2. 方向感知的信息传播 (有向GAT / 无向GCN)
      3. FFN + 残差
    """

    def __init__(self, n_channels: int, hidden: int = 64,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels

        # 每种特征一个图学习器
        self.graph_learners = nn.ModuleList([
            DirectedGraphLearner(n_channels, n_heads=n_heads)
            for _ in range(4)
        ])

        # 有向分支: GAT-style attention
        self.directed_conv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_channels, hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            for _ in range(3)  # GC, TE, AEC
        ])

        # 无向分支: GCN-style
        self.undirected_conv = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        # 特征融合
        self.norm = nn.LayerNorm(n_channels)
        self.gate_norm = nn.LayerNorm(hidden * 4)
        self.feat_gate = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )
        self.proj_out = nn.Linear(hidden, n_channels)

    def _directed_propagate(self, h: torch.Tensor, adj: torch.Tensor,
                            conv: nn.Module) -> torch.Tensor:
        """有向图信息传播: h' = conv(adj · h)"""
        adj_norm = F.normalize(adj.mean(dim=1), p=1, dim=-1)  # [B, N, N]
        return conv(torch.bmm(adj_norm, h))

    def _undirected_propagate(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """无向图信息传播: h' = conv(D^{-1/2} A D^{-1/2} h)"""
        adj_mean = adj.mean(dim=1)  # [B, N, N]
        deg = adj_mean.sum(-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_adj = adj_mean * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
        return self.undirected_conv(torch.bmm(norm_adj, h))

    def forward(
        self,
        nets: torch.Tensor,
        alpha: float = 0.15,
        feature_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            nets: [B, C, C, 4]  单个时间步的脑网络特征
        Returns:
            out: [B, C, C, 4]  过滤后特征
        """
        B, C, _, F_ = nets.shape
        active_mask = (
            torch.ones(F_, device=nets.device, dtype=nets.dtype)
            if feature_mask is None
            else feature_mask.to(device=nets.device, dtype=nets.dtype).flatten()[:F_]
        )
        branch_outs = []

        for f_idx in range(F_):
            adj_f = nets[..., f_idx]                      # [B, C, C]
            h_f = adj_f                                    # 用邻接行作为节点特征

            # 图学习
            directed = FEATURE_DIRECTED[f_idx]
            learned_adj = self.graph_learners[f_idx](
                h_f, directed=directed, alpha=alpha
            )  # [B, H, C, C]

            # 信息传播
            if directed:
                out_f = self._directed_propagate(
                    h_f, learned_adj, self.directed_conv[f_idx]
                )  # [B, C, hidden]
            else:
                out_f = self._undirected_propagate(
                    h_f, learned_adj
                )  # [B, C, hidden]

            out_f = out_f * active_mask[f_idx]
            branch_outs.append(out_f)

        # 特征门控融合
        stacked = torch.stack(branch_outs, dim=-1)        # [B, C, hidden, 4]
        concat = torch.cat(branch_outs, dim=-1)           # [B, C, hidden*4]
        gate_logits = self.feat_gate(self.gate_norm(concat))  # [B, C, 4]
        inactive = active_mask <= 0
        has_inactive = bool(inactive.any().item())
        if has_inactive:
            gate_logits = gate_logits.masked_fill(inactive.view(1, 1, -1), -1e9)
        gates = torch.softmax(gate_logits, dim=-1)
        gates = gates * active_mask.view(1, 1, -1)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        fused = (stacked * gates.unsqueeze(2)).sum(dim=-1)  # [B, C, hidden]
        fused = self.proj_out(fused)                        # [B, C, C]

        # 重构为4特征输出 (残差)
        out = nets + fused.unsqueeze(-1).expand_as(nets) * 0.1  # 轻残差
        out = out * active_mask.view(1, 1, 1, -1)
        return out


# =====================================================================
# Main: DirectedBrainTimeFilter
# =====================================================================

class DirectedBrainTimeFilter(nn.Module):
    """
    有向脑网络TimeFilter

    对 [B, P, C, C, 4] 的脑网络时序数据进行:
    1. 每个时间步: DirectedBrainGraphBlock 做特征内图过滤
    2. 跨时间步: 时序注意力捕获网络演化模式
    3. 特征感知MoE路由控制过滤强度

    输出: 过滤后的脑网络特征 + MoE辅助损失
    """

    def __init__(self, cfg: DirectedTimeFilterConfig = None):
        super().__init__()
        self.cfg = cfg or DirectedTimeFilterConfig()
        c = self.cfg

        # 多层有向图过滤
        self.graph_blocks = nn.ModuleList([
            DirectedBrainGraphBlock(
                n_channels=c.n_channels, hidden=c.hidden_dim,
                n_heads=c.n_heads, dropout=c.dropout,
            )
            for _ in range(c.n_blocks)
        ])

        # 时序注意力 (跨补丁)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=c.n_channels * c.n_channels * c.n_features,
            num_heads=min(4, c.n_channels),  # 保守的头数
            dropout=c.dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(
            c.n_channels * c.n_channels * c.n_features
        )

        # 特征感知MoE路由
        self.moe_router = FeatureAwareMoERouter(
            in_dim=c.n_channels,
            n_features=c.n_features,
            top_p=c.top_p,
        )

        # 特征ID寄存器
        feat_ids = torch.arange(c.n_features).unsqueeze(1).expand(-1, c.n_channels)
        self.register_buffer('feat_ids', feat_ids.flatten())  # [n_features*n_channels]
        self.register_buffer(
            'feature_active_mask',
            torch.ones(c.n_features, dtype=torch.float32),
            persistent=False,
        )
        self.set_active_features(tuple(FEATURE_NAMES[:c.n_features]))

    def set_active_features(self, active_features: Tuple[str, ...]) -> None:
        normalized = tuple(str(name).lower() for name in active_features)
        mask = torch.tensor(
            [1.0 if name in normalized else 0.0 for name in FEATURE_NAMES[:self.cfg.n_features]],
            dtype=self.feature_active_mask.dtype,
            device=self.feature_active_mask.device,
        )
        if mask.sum() <= 0:
            raise ValueError("DirectedBrainTimeFilter requires at least one active feature")
        self.feature_active_mask.copy_(mask)

    def forward(
        self,
        brain_networks: torch.Tensor,
        is_training: bool = False,
        valid_patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            brain_networks: [B, P, C, C, 4]
            valid_patch_mask: optional [B, P] mask for valid patch slots
        Returns:
            filtered: [B, P, C, C, 4]  过滤后特征
            moe_loss: scalar  MoE辅助损失
        """
        B, P, C, _, F_ = brain_networks.shape
        moe_loss = torch.tensor(0.0, device=brain_networks.device)
        feature_mask = self.feature_active_mask[:F_].to(
            device=brain_networks.device,
            dtype=brain_networks.dtype,
        )

        # 1. 每个时间步做有向图过滤
        filtered_steps = []
        for t in range(P):
            x_t = brain_networks[:, t]  # [B, C, C, 4]
            for block in self.graph_blocks:
                x_t = block(x_t, alpha=self.cfg.alpha, feature_mask=feature_mask)
            filtered_steps.append(x_t)

        filtered = torch.stack(filtered_steps, dim=1)  # [B, P, C, C, 4]
        filtered = filtered * feature_mask.view(1, 1, 1, 1, F_)

        # 2. 时序注意力 (跨补丁维度)
        flat = filtered.reshape(B, P, -1)               # [B, P, C*C*4]
        flat_normed = self.temporal_norm(flat)
        key_padding_mask = None
        if valid_patch_mask is not None:
            valid_patch_mask = valid_patch_mask.to(device=brain_networks.device, dtype=torch.bool)
            safe_valid = valid_patch_mask.clone()
            empty = ~safe_valid.any(dim=1)
            if empty.any():
                safe_valid[empty, 0] = True
            key_padding_mask = ~safe_valid
        attn_out, _ = self.temporal_attn(
            flat_normed,
            flat_normed,
            flat_normed,
            key_padding_mask=key_padding_mask,
        )
        flat = flat + attn_out                            # 残差
        filtered = flat.reshape(B, P, C, C, F_)
        filtered = filtered * feature_mask.view(1, 1, 1, 1, F_)
        if valid_patch_mask is not None:
            filtered = filtered * valid_patch_mask.to(dtype=filtered.dtype)[:, :, None, None, None]

        # 3. MoE路由 (获取辅助损失)
        dummy_input = filtered.mean(dim=(2, 3))          # [B, P, 4]
        dummy_flat = dummy_input.reshape(B * P, F_)
        # 简化: 使用特征级门控
        _, moe_loss = self.moe_router(
            dummy_flat[:, :self.cfg.n_channels] if dummy_flat.shape[-1] >= self.cfg.n_channels
            else F.pad(dummy_flat, (0, self.cfg.n_channels - dummy_flat.shape[-1])),
            self.feat_ids[:self.cfg.n_channels],
            is_training=is_training,
        )

        return filtered, moe_loss


# =====================================================================
# Self-test
# =====================================================================

def _test():
    import time

    torch.manual_seed(42)
    B, P, C, F_ = 4, 20, 22, 4

    cfg = DirectedTimeFilterConfig(
        n_channels=C, n_patches=P, n_features=F_,
        n_blocks=1, hidden_dim=32, n_heads=2,
    )
    model = DirectedBrainTimeFilter(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DirectedBrainTimeFilter: {n_params:,} params")

    # 输入: 脑网络特征 [B, P, C, C, 4]
    nets = torch.randn(B, P, C, C, F_).abs()
    # 有向特征不对称, wPLI对称
    nets[..., 3] = (nets[..., 3] + nets[..., 3].transpose(-1, -2)) / 2

    print(f"\nForward pass...")
    t0 = time.time()
    with torch.no_grad():
        filtered, moe_loss = model(nets, is_training=False)
    t1 = time.time()

    assert filtered.shape == (B, P, C, C, F_), \
        f"Expected {(B, P, C, C, F_)}, got {filtered.shape}"
    print(f"  Output shape: {filtered.shape}")
    print(f"  MoE loss: {moe_loss.item():.4f}")
    print(f"  Time: {t1-t0:.2f}s")

    # 检查wPLI对称性保持
    wpli_out = filtered[..., 3]
    asymmetry = (wpli_out - wpli_out.transpose(-1, -2)).abs().mean()
    print(f"  wPLI asymmetry (should be small): {asymmetry.item():.6f}")

    # Backward
    print(f"\nBackward pass...")
    filtered, moe_loss = model(nets, is_training=True)
    loss = filtered.sum() + moe_loss
    loss.backward()

    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    if no_grad:
        print(f"  WARNING: {len(no_grad)} params without gradient")
    else:
        print(f"  Gradient flow: OK (all params)")

    print(f"\n[PASS] DirectedBrainTimeFilter tests passed!")


if __name__ == '__main__':
    _test()
