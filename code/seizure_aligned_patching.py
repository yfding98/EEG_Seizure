#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeizureAlignedAdaptivePatching — 发作对齐的自适应补丁划分

以发作起始时刻 (seizure onset) 为锚点进行补丁划分，确保发作瞬间
始终落在第 n_pre / (n_pre+1) 个补丁的交界处。

补丁网格：
    ┌───────── 发作前 (4 s) ─────────┐┌───────────── 发作后 (6 s) ──────────────┐
    │ P0  P1  P2  P3  P4  P5  P6  P7 ││ P8  P9  P10 P11 P12 P13 P14 … P19     │
    └─────────────────────────────────┘└─────────────────────────────────────────┘
                                      ↑
                              seizure onset (锚点)

若发作起始距窗口边界不足，自动减少该侧补丁数，不做信号级零填充。
输出张量在 batch 维按 max_patches 对齐，无效 patch 位补零。
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn


class SeizureAlignedAdaptivePatching(nn.Module):
    """
    发作对齐的自适应补丁划分

    Parameters
    ----------
    n_channels : int
        输入通道数 (默认 22 TCP 双极导联)
    patch_len : int
        每个补丁的采样点数 (默认 100 = 0.5 s @ 200 Hz)
    n_pre_patches : int
        发作前的最大补丁数 (默认 8 → 4 s)
    n_post_patches : int
        发作后的最大补丁数 (默认 12 → 6 s)
    fs : float
        采样率 (默认 200.0 Hz)
    """

    def __init__(
        self,
        n_channels: int = 22,
        patch_len: int = 100,
        n_pre_patches: int = 8,
        n_post_patches: int = 12,
        fs: float = 200.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.n_pre_patches = n_pre_patches
        self.n_post_patches = n_post_patches
        self.max_patches = n_pre_patches + n_post_patches  # 20
        self.fs = fs
        self.patch_sec = patch_len / fs  # 0.5 s

        # 补丁 slot 相对于 onset 的采样点偏移量
        #   slot 0 → -8 * 100 = -800,  slot 7 → -1 * 100 = -100
        #   slot 8 →  0 * 100 =    0,  slot 19 → 11 * 100 = 1100
        offsets = torch.arange(-n_pre_patches, n_post_patches) * patch_len
        self.register_buffer('_offsets', offsets)  # [max_patches]

        # 每个 slot 相对 onset 的时间偏移 (秒)
        rel_time = offsets.float() / fs
        self.register_buffer('_rel_time', rel_time)  # [max_patches]

        # forward() 后缓存的元数据 (用于 get_patch_timestamps)
        self._window_start_sec: Optional[torch.Tensor] = None
        self._seizure_onset_sec: Optional[torch.Tensor] = None
        self._valid_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        seizure_onset_sec: torch.Tensor,
        window_start_sec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        x : [B, C, T]
            22 通道双极导联原始信号 (200 Hz)
        seizure_onset_sec : [B]
            每个样本的发作起始绝对时间 (秒)
        window_start_sec : [B]
            每个样本的窗口起始绝对时间 (秒)

        Returns
        -------
        patches : [B, max_patches, C, patch_len]
            补丁张量。无效位补零。
        valid_patch_counts : [B]  (long)
            每个样本的有效补丁数
        seizure_relative_time : [B, max_patches]
            每个补丁起始时刻相对发作 onset 的时间 (秒，可为负)
        """
        B, C, T = x.shape
        device = x.device
        P = self.max_patches
        L = self.patch_len

        # ── 1. 发作在窗口中的采样点索引 ──
        onset_sample = (
            (seizure_onset_sec - window_start_sec) * self.fs
        ).long()  # [B]

        # ── 2. 每个 patch slot 的起始采样点 ──
        #       starts[b, k] = onset_sample[b] + offsets[k]
        starts = onset_sample.unsqueeze(1) + self._offsets.unsqueeze(0)  # [B, P]

        # ── 3. 有效性判断 ──
        #       补丁完全在 [0, T) 范围内才有效
        valid = (starts >= 0) & (starts + L <= T)  # [B, P]

        # ── 4. 安全索引 (无效 slot 指向 [0:L]，最后清零) ──
        clamp_max = max(T - L, 0)
        starts_safe = starts.clamp(0, clamp_max)  # [B, P]

        # ── 5. 展开为逐采样点索引 [B, P, L] ──
        sample_offsets = torch.arange(L, device=device)  # [L]
        patch_indices = starts_safe.unsqueeze(-1) + sample_offsets  # [B, P, L]

        # ── 6. gather 提取补丁 (纯向量化，无循环) ──
        #       x: [B, C, T]  →  patches: [B, C, P, L]  →  permute → [B, P, C, L]
        idx = patch_indices.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, P, L]
        idx_flat = idx.reshape(B, C, P * L)                      # [B, C, P*L]
        gathered = torch.gather(x, 2, idx_flat)                   # [B, C, P*L]
        patches = gathered.reshape(B, C, P, L).permute(0, 2, 1, 3)  # [B, P, C, L]

        # ── 7. 清零无效 slot ──
        mask_4d = valid.unsqueeze(-1).unsqueeze(-1).float()  # [B, P, 1, 1]
        patches = patches * mask_4d

        # ── 8. 有效补丁计数 ──
        valid_patch_counts = valid.long().sum(dim=1)  # [B]

        # ── 9. 相对时间 (每个 slot 相对 onset 的秒数, 无效位也保留便于对齐) ──
        seizure_relative_time = self._rel_time.unsqueeze(0).expand(B, -1)  # [B, P]

        # ── 缓存元数据 ──
        self._window_start_sec = window_start_sec.detach()
        self._seizure_onset_sec = seizure_onset_sec.detach()
        self._valid_mask = valid.detach()

        return patches, valid_patch_counts, seizure_relative_time

    # ------------------------------------------------------------------
    # get_patch_timestamps
    # ------------------------------------------------------------------

    def get_patch_timestamps(self, batch_idx: int) -> torch.Tensor:
        """
        获取指定样本每个有效补丁的绝对起始时间戳。

        Parameters
        ----------
        batch_idx : int
            batch 中的样本索引

        Returns
        -------
        timestamps : [n_valid]
            有效补丁的绝对起始时间 (秒)

        Raises
        ------
        RuntimeError
            若尚未调用 forward()
        """
        if self._seizure_onset_sec is None:
            raise RuntimeError("请先调用 forward() 后再获取时间戳")

        onset = self._seizure_onset_sec[batch_idx]
        valid = self._valid_mask[batch_idx]  # [max_patches]

        # 每个 slot 的绝对起始时间
        all_times = onset + self._rel_time  # [max_patches]

        return all_times[valid]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def get_valid_mask(self) -> Optional[torch.Tensor]:
        """返回最近一次 forward 的有效 patch mask [B, max_patches]"""
        return self._valid_mask

    def extra_repr(self) -> str:
        return (
            f"n_channels={self.n_channels}, "
            f"patch_len={self.patch_len} ({self.patch_sec:.2f}s), "
            f"pre={self.n_pre_patches}, post={self.n_post_patches}, "
            f"max_patches={self.max_patches}, fs={self.fs}"
        )


# ======================================================================
# 单元测试
# ======================================================================

def _test():
    """基本功能验证"""
    torch.manual_seed(42)

    B, C, T = 4, 22, 2400          # 12 秒 @ 200 Hz
    fs = 200.0

    module = SeizureAlignedAdaptivePatching(
        n_channels=C, patch_len=100, n_pre_patches=8,
        n_post_patches=12, fs=fs,
    )
    print(module)

    x = torch.randn(B, C, T)

    # ---- Case 1: onset 在窗口中间 (5s 处) ──
    seizure_onset = torch.tensor([105.0, 105.0, 102.0, 109.0])
    window_start  = torch.tensor([100.0, 100.0, 100.0, 100.0])

    patches, counts, rel_time = module(x, seizure_onset, window_start)

    print(f"\npatches      : {patches.shape}")   # [4, 20, 22, 100]
    print(f"valid_counts : {counts}")
    print(f"rel_time[0]  : {rel_time[0].tolist()}")

    # Sample 0: onset=5s → 1000th sample
    #   pre:  1000 // 100 = 10, capped at 8 → 8 patches
    #   post: (2400-1000) // 100 = 14, capped at 12 → 12 patches
    #   total = 20
    assert counts[0].item() == 20, f"Expected 20, got {counts[0].item()}"

    # Sample 2: onset=2s → 400th sample
    #   pre:  400 // 100 = 4, capped at 4 → 4 patches (不足8)
    #   post: (2400-400) // 100 = 20, capped at 12 → 12 patches
    #   total = 16
    assert counts[2].item() == 16, f"Expected 16, got {counts[2].item()}"

    # Sample 3: onset=9s → 1800th sample
    #   pre:  1800 // 100 = 18, capped at 8 → 8 patches
    #   post: (2400-1800) // 100 = 6 → 6 patches (不足12)
    #   total = 14
    assert counts[3].item() == 14, f"Expected 14, got {counts[3].item()}"

    # ---- 发作对齐验证: onset 恰好在 patch 7/8 交界 ──
    # patches[0, 7] = x[0, :, 900:1000]  (onset-100 ~ onset)
    # patches[0, 8] = x[0, :, 1000:1100] (onset ~ onset+100)
    onset_idx = 1000
    expected_pre = x[0, :, onset_idx - 100: onset_idx]    # [22, 100]
    expected_post = x[0, :, onset_idx: onset_idx + 100]   # [22, 100]
    assert torch.allclose(patches[0, 7], expected_pre), "Pre-onset patch mismatch"
    assert torch.allclose(patches[0, 8], expected_post), "Post-onset patch mismatch"

    # ---- 无效 patch 应全零 ──
    # Sample 2: slots 0-3 应为零 (only 4 pre-patches)
    assert patches[2, 0].abs().sum() == 0, "Invalid patch should be zero"
    assert patches[2, 3].abs().sum() == 0, "Invalid patch should be zero"
    assert patches[2, 4].abs().sum() > 0, "Valid patch should be non-zero"

    # ---- get_patch_timestamps ──
    ts = module.get_patch_timestamps(0)
    print(f"\ntimestamps[0]: {ts.tolist()}")
    assert len(ts) == 20
    assert abs(ts[8].item() - 105.0) < 1e-5, "Onset patch should align with seizure_onset_sec"

    print("\n[PASS] All tests passed!")


if __name__ == '__main__':
    _test()
