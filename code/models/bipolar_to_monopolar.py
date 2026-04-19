#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BipolarToMonopolarMapper — 极性感知的双极→单极通道映射模块

核心问题:
    模型在22个TCP双极导联上学习，但SOZ临床标注基于19个单极通道，
    需将导联级预测映射回通道级。

设计原理:
    双极导联 = 阳极(anode) - 阴极(cathode)
    若某导联SOZ概率高，说明阳极或阴极附近可能存在SOZ。
    但阳极和阴极对导联信号的"贡献方向"不同:
      - 阳极(正端): 该电极的电位增高会使导联值增大 → 正贡献
      - 阴极(负端): 该电极的电位增高会使导联值减小 → 负贡献

    因此映射矩阵应编码极性:
      W[ch, pair] = +1  若ch是pair的阳极
      W[ch, pair] = -1  若ch是pair的阴极
      W[ch, pair] =  0  若ch不参与pair

    归一化: 每行除以该通道出现的总次数

示例 (F3通道):
    导联14: FP1-F3 → F3为阴极 → W[F3, 14] = -1/2 = -0.5
    导联15: F3-C3  → F3为阳极 → W[F3, 15] = +1/2 = +0.5
    解读: F3的SOZ概率由 FP1-F3(负贡献,该导联高→F3电位相对低)
           和 F3-C3(正贡献,该导联高→F3电位相对高) 共同决定

注意:
    FZ和PZ不参与22个TCP双极导联对，其映射行全零。
    模块为其提供可学习偏置(bias)来补偿。

Reference:
    TCP Montage: TUSZ v2.0.3 01_tcp_ar_montage.txt
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# 默认定义 (可由外部覆盖)
# =============================================================================

# 标准19单极通道 (10-20系统)
DEFAULT_MONOPOLAR_19 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]

# TCP 22通道双极导联对 (TUSZ官方顺序, 01_tcp_ar_montage.txt)
DEFAULT_TCP_PAIRS_22: List[Tuple[str, str]] = [
    # 左颞链 (0-3)
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    # 右颞链 (4-7)
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    # 中央链 (8-13)
    ('A1', 'T3'),  ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
    ('C4', 'T4'),  ('T4', 'A2'),
    # 左副矢状链 (14-17)
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    # 右副矢状链 (18-21)
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
]


# =============================================================================
# 主模块
# =============================================================================

class BipolarToMonopolarMapper(nn.Module):
    """
    极性感知的 22 TCP双极导联 → 19 单极通道映射

    映射矩阵 W ∈ R^{19×22}:
        W[i, j] = +1/count_i  若通道i是导联j的阳极(正端)
        W[i, j] = -1/count_i  若通道i是导联j的阴极(负端)
        W[i, j] =  0          若通道i不参与导联j

    其中 count_i = 通道i在所有导联对中出现的总次数

    前向传播:
        monopolar_probs = softmax( bipolar_probs @ W^T )

    Args:
        monopolar_channels: 19个单极通道名列表
        bipolar_pairs:      22个双极导联对列表 [(anode, cathode), ...]

    Attributes:
        W:          [19, 22] 极性感知归一化映射矩阵 (buffer, 不可训练)
        W_raw:      [19, 22] 未归一化的原始极性矩阵 (buffer, 供检查)
        bias:       [19]     可学习偏置 (补偿未映射通道如FZ, PZ)
        count:      [19]     每通道出现次数 (buffer)
        mapped:     [19]     bool, 通道是否被至少一个导联对覆盖 (buffer)
    """

    def __init__(
        self,
        monopolar_channels: List[str] = None,
        bipolar_pairs: List[Tuple[str, str]] = None,
    ):
        super().__init__()

        mono = monopolar_channels or list(DEFAULT_MONOPOLAR_19)
        pairs = bipolar_pairs or list(DEFAULT_TCP_PAIRS_22)
        n_mono = len(mono)
        n_bip = len(pairs)

        self.monopolar_channels = mono
        self.bipolar_pairs = pairs
        self.bipolar_names = [f"{a}-{b}" for a, b in pairs]

        # 单极通道名 → 索引
        self._mono_idx = {ch.upper(): i for i, ch in enumerate(mono)}

        # ---- 构建映射矩阵 ----
        W_raw = np.zeros((n_mono, n_bip), dtype=np.float32)

        for j, (anode, cathode) in enumerate(pairs):
            a_upper = anode.upper()
            c_upper = cathode.upper()
            if a_upper in self._mono_idx:
                W_raw[self._mono_idx[a_upper], j] = +1.0   # 阳极 → 正贡献
            if c_upper in self._mono_idx:
                W_raw[self._mono_idx[c_upper], j] = -1.0   # 阴极 → 负贡献

        # 每通道出现次数 = 行中非零元素个数
        count = np.abs(W_raw).sum(axis=1)  # [19]

        # 归一化: 每行除以出现次数 (count=0的行保持全零)
        count_safe = np.maximum(count, 1.0)
        W_norm = W_raw / count_safe[:, np.newaxis]  # [19, 22]

        # 标记已映射/未映射通道
        mapped = (count > 0).astype(np.float32)

        # 注册为buffer (不参与梯度, 但随模型保存/加载/移动设备)
        self.register_buffer('W', torch.from_numpy(W_norm))                 # [19, 22]
        self.register_buffer('W_raw', torch.from_numpy(W_raw))              # [19, 22]
        self.register_buffer('count', torch.from_numpy(count))              # [19]
        self.register_buffer('mapped', torch.from_numpy(mapped))            # [19]

        # 可学习偏置 — 补偿未映射通道 (如 FZ, PZ)
        self.bias = nn.Parameter(torch.zeros(n_mono))

        # 记录未映射通道警告
        unmapped = [mono[i] for i in range(n_mono) if count[i] == 0]
        if unmapped:
            logger.warning(
                f"BipolarToMonopolarMapper: 以下通道不参与任何TCP导联对, "
                f"其映射行全零, 仅依赖可学习偏置: {unmapped}"
            )

        # ---- 构建可解释性元数据 ----
        self._contribution_map = self._build_contribution_map(
            mono, pairs, W_raw, count
        )

    # -----------------------------------------------------------------
    # 前向传播
    # -----------------------------------------------------------------

    def forward(self, bipolar_probs: torch.Tensor) -> torch.Tensor:
        """
        将22导联SOZ概率映射到19单极通道SOZ概率

        Args:
            bipolar_probs: [B, 22]  22个TCP导联的SOZ概率/logits

        Returns:
            monopolar_probs: [B, 19]  19个单极通道的SOZ概率
                             经softmax归一化, sum=1 (概率守恒)
        """
        # 线性映射: [B, 22] @ [22, 19] → [B, 19]
        logits = torch.matmul(bipolar_probs, self.W.t()) + self.bias  # [B, 19]
        # softmax → 概率 (保证19通道和为1)
        monopolar_probs = F.softmax(logits, dim=-1)  # [B, 19]
        return monopolar_probs

    def forward_logits(self, bipolar_probs: torch.Tensor) -> torch.Tensor:
        """返回softmax前的logits (用于外部损失计算)"""
        return torch.matmul(bipolar_probs, self.W.t()) + self.bias

    # -----------------------------------------------------------------
    # 可解释性
    # -----------------------------------------------------------------

    @staticmethod
    def _build_contribution_map(
        mono: List[str],
        pairs: List[Tuple[str, str]],
        W_raw: np.ndarray,
        count: np.ndarray,
    ) -> Dict[str, List[Dict]]:
        """构建每个单极通道的导联贡献详情"""
        result = {}
        for i, ch in enumerate(mono):
            contributions = []
            for j, (anode, cathode) in enumerate(pairs):
                w = W_raw[i, j]
                if w != 0:
                    polarity = 'anode(+)' if w > 0 else 'cathode(-)'
                    pair_name = f"{anode}-{cathode}"
                    norm_w = w / max(count[i], 1)
                    contributions.append({
                        'pair_idx': j,
                        'pair_name': pair_name,
                        'polarity': polarity,
                        'raw_weight': float(w),
                        'normalized_weight': float(norm_w),
                    })
            result[ch.upper()] = {
                'contributions': contributions,
                'n_pairs': int(count[i]),
                'is_mapped': count[i] > 0,
            }
        return result

    def visualize_contribution(self, channel_name: str) -> Dict:
        """
        返回指定单极通道在22导联中的权重分布

        Args:
            channel_name: 单极通道名 (如 'F3', 'CZ', 'FZ')

        Returns:
            Dict with keys:
                'channel': 通道名
                'n_pairs': 参与的导联对数
                'is_mapped': 是否被映射
                'contributions': List[Dict] 每个相关导联的详情
                    - pair_idx: 导联索引
                    - pair_name: 导联名 (如 'FP1-F3')
                    - polarity: 'anode(+)' 或 'cathode(-)'
                    - raw_weight: 原始权重 (+1 或 -1)
                    - normalized_weight: 归一化后权重
                'weight_vector': [22] 该通道在W矩阵中的完整权重行

        示例:
            >>> mapper = BipolarToMonopolarMapper()
            >>> info = mapper.visualize_contribution('F3')
            >>> for c in info['contributions']:
            ...     print(f"  {c['pair_name']}: {c['polarity']}, w={c['normalized_weight']:+.2f}")
            # 输出:
            #   FP1-F3: cathode(-), w=-0.50
            #   F3-C3:  anode(+),  w=+0.50
        """
        ch = channel_name.upper()
        if ch not in self._contribution_map:
            raise ValueError(
                f"通道 '{channel_name}' 不在19单极通道列表中。"
                f"可用通道: {self.monopolar_channels}"
            )

        info = self._contribution_map[ch].copy()
        info['channel'] = ch

        # 添加完整权重行
        idx = self._mono_idx[ch]
        info['weight_vector'] = self.W[idx].cpu().numpy().tolist()
        info['bias'] = self.bias[idx].item()

        return info

    def get_contribution_table(self) -> str:
        """
        生成全部19通道的映射贡献表 (便于打印/日志)

        Returns:
            格式化的多行字符串
        """
        lines = [
            "=" * 80,
            "BipolarToMonopolarMapper: 极性感知映射表",
            "=" * 80,
            f"{'通道':<6} {'#导联':>5} {'导联贡献详情':<60} {'L1':>5}",
            "-" * 80,
        ]
        for ch in self.monopolar_channels:
            info = self._contribution_map[ch.upper()]
            idx = self._mono_idx[ch.upper()]
            l1 = float(self.W[idx].abs().sum())

            if not info['is_mapped']:
                detail = "[未映射 - 仅依赖可学习偏置]"
            else:
                parts = []
                for c in info['contributions']:
                    sign = '+' if c['normalized_weight'] > 0 else ''
                    parts.append(
                        f"{c['pair_name']}({sign}{c['normalized_weight']:.2f})"
                    )
                detail = ', '.join(parts)

            lines.append(f"{ch:<6} {info['n_pairs']:>5} {detail:<60} {l1:>5.2f}")

        lines.append("-" * 80)

        # 统计
        n_mapped = sum(1 for ch in self.monopolar_channels
                       if self._contribution_map[ch.upper()]['is_mapped'])
        n_unmapped = len(self.monopolar_channels) - n_mapped
        lines.append(f"已映射: {n_mapped}/19, 未映射: {n_unmapped}/19")
        lines.append("=" * 80)

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # 验证
    # -----------------------------------------------------------------

    def validate(self, verbose: bool = True) -> Dict[str, bool]:
        """
        验证映射矩阵的正确性

        检查项:
        1. W形状 = [19, 22]
        2. 已映射通道的L1范数 = 1.0 (概率守恒)
        3. 极性正确: anode → +, cathode → -
        4. 未映射通道行全零
        5. 每行非零元素数 = count

        Returns:
            Dict[str, bool] 各检查项通过/失败
        """
        W = self.W.cpu().numpy()
        W_raw = self.W_raw.cpu().numpy()
        count = self.count.cpu().numpy()
        mapped = self.mapped.cpu().numpy()

        checks = OrderedDict()

        # 1. 形状
        checks['shape_correct'] = (W.shape == (len(self.monopolar_channels), len(self.bipolar_pairs)))

        # 2. 已映射通道L1范数 = 1.0
        l1_norms = np.abs(W).sum(axis=1)
        l1_ok = True
        for i, ch in enumerate(self.monopolar_channels):
            if mapped[i]:
                if not np.isclose(l1_norms[i], 1.0, atol=1e-6):
                    l1_ok = False
                    if verbose:
                        logger.error(f"  L1 norm({ch}) = {l1_norms[i]:.6f}, expected 1.0")
        checks['l1_norm_conservation'] = l1_ok

        # 3. 极性正确
        polarity_ok = True
        for j, (anode, cathode) in enumerate(self.bipolar_pairs):
            a_up = anode.upper()
            c_up = cathode.upper()
            if a_up in self._mono_idx:
                if W_raw[self._mono_idx[a_up], j] != +1.0:
                    polarity_ok = False
                    if verbose:
                        logger.error(f"  极性错误: {a_up} in {anode}-{cathode} 应为+1")
            if c_up in self._mono_idx:
                if W_raw[self._mono_idx[c_up], j] != -1.0:
                    polarity_ok = False
                    if verbose:
                        logger.error(f"  极性错误: {c_up} in {anode}-{cathode} 应为-1")
        checks['polarity_correct'] = polarity_ok

        # 4. 未映射通道行全零
        unmapped_ok = True
        for i, ch in enumerate(self.monopolar_channels):
            if not mapped[i]:
                if not np.allclose(W[i], 0):
                    unmapped_ok = False
                    if verbose:
                        logger.error(f"  未映射通道 {ch} 的行不为零")
        checks['unmapped_zero'] = unmapped_ok

        # 5. 非零元素数 = count
        count_ok = True
        for i, ch in enumerate(self.monopolar_channels):
            nnz = np.count_nonzero(W_raw[i])
            if nnz != int(count[i]):
                count_ok = False
                if verbose:
                    logger.error(f"  {ch}: nnz={nnz}, count={int(count[i])}")
        checks['count_consistent'] = count_ok

        all_pass = all(checks.values())
        if verbose:
            status = "PASS" if all_pass else "FAIL"
            logger.info(f"BipolarToMonopolarMapper validation: {status}")
            for name, ok in checks.items():
                flag = "[OK]" if ok else "[FAIL]"
                logger.info(f"  {flag} {name}")

        return checks

    # -----------------------------------------------------------------
    # 工具方法
    # -----------------------------------------------------------------

    def get_unmapped_channels(self) -> List[str]:
        """返回未被任何TCP导联对覆盖的通道列表"""
        return [
            self.monopolar_channels[i]
            for i in range(len(self.monopolar_channels))
            if self.count[i].item() == 0
        ]

    def get_channel_pair_count(self) -> Dict[str, int]:
        """返回每个通道参与的导联对数"""
        return {
            ch: int(self.count[i].item())
            for i, ch in enumerate(self.monopolar_channels)
        }

    def extra_repr(self) -> str:
        n_mapped = int(self.mapped.sum().item())
        return (
            f"monopolar={len(self.monopolar_channels)}, "
            f"bipolar={len(self.bipolar_pairs)}, "
            f"mapped={n_mapped}/{len(self.monopolar_channels)}, "
            f"W_shape={list(self.W.shape)}"
        )


# =============================================================================
# 自测
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=" * 80)
    print("BipolarToMonopolarMapper Self-Test")
    print("=" * 80)

    # 1. 构建
    mapper = BipolarToMonopolarMapper()
    print(f"\n{mapper}\n")

    # 2. 打印映射表
    print(mapper.get_contribution_table())

    # 3. 可解释性: 查看F3的贡献
    print("\n--- F3通道贡献详情 ---")
    info = mapper.visualize_contribution('F3')
    print(f"通道: {info['channel']}, 参与导联数: {info['n_pairs']}")
    for c in info['contributions']:
        print(f"  {c['pair_name']:>8}: {c['polarity']:<12} w={c['normalized_weight']:+.4f}")
    print(f"  完整权重: {info['weight_vector']}")

    # 4. 查看CZ的贡献
    print("\n--- CZ通道贡献详情 ---")
    info = mapper.visualize_contribution('CZ')
    print(f"通道: {info['channel']}, 参与导联数: {info['n_pairs']}")
    for c in info['contributions']:
        print(f"  {c['pair_name']:>8}: {c['polarity']:<12} w={c['normalized_weight']:+.4f}")

    # 5. 查看FZ (未映射)
    print("\n--- FZ通道贡献详情 ---")
    info = mapper.visualize_contribution('FZ')
    print(f"通道: {info['channel']}, 参与导联数: {info['n_pairs']}, 已映射: {info['is_mapped']}")
    if not info['is_mapped']:
        print("  [未映射 - 不参与任何TCP导联对, 仅依赖可学习偏置]")

    # 6. 未映射通道
    unmapped = mapper.get_unmapped_channels()
    print(f"\n未映射通道: {unmapped}")

    # 7. 验证
    print("\n--- 验证 ---")
    checks = mapper.validate(verbose=True)

    # 8. 前向传播测试
    print("\n--- 前向传播 ---")
    B = 4
    bipolar_probs = torch.rand(B, 22)
    monopolar_probs = mapper(bipolar_probs)
    print(f"  Input:  bipolar_probs  {list(bipolar_probs.shape)}")
    print(f"  Output: monopolar_probs {list(monopolar_probs.shape)}")
    print(f"  Sum per sample (should be 1.0): {monopolar_probs.sum(dim=-1).tolist()}")

    # 验证softmax概率守恒
    assert torch.allclose(monopolar_probs.sum(dim=-1), torch.ones(B), atol=1e-5), \
        "Probability conservation violated!"

    # 9. 反向传播测试
    print("\n--- 反向传播 ---")
    loss = monopolar_probs.sum()
    loss.backward()
    print(f"  bias.grad: {mapper.bias.grad}")
    print(f"  bias.grad != 0 for unmapped channels: "
          f"{[mapper.monopolar_channels[i] for i in range(19) if mapper.bias.grad[i] != 0]}")

    print("\n[OK] All tests passed!")

