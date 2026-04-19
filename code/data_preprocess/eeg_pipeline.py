#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG癫痫数据预处理管道 — TimeFilter SOZ检测

输入:
    EDF文件路径列表 + 临床标注CSV (发作起始时间, SOZ通道)

输出:
    X:        [batch, 22, 20, 100]   22 TCP导联 × 20补丁 × 100采样点
    y_soz:    [batch, 19]            19通道SOZ二值标签
    metadata: list[dict]             含 data_source='public'/'private'

处理流程:
    a) 读取EDF → 提取21电极 (标准19 + A1/A2)
    b) MNE带通滤波 3-45Hz (Butterworth 4阶, 零相位)
    c) MNE重采样 → 200Hz
    d) 提取发作窗口 [onset-5s, onset+5s], 坏段检测 (>500μV / 平坦>1s) → 剔除
    e) 幅值裁剪 ±1 std (基于基线期=发作前5s计算)
    f) 21ch → 22 TCP双极导联
    g) 基线标准化 (z-score, 基于双极基线期)
    h) 分割为补丁: 10s=2000点 → 20补丁 × 100点

TCP 22通道双极导联 (按用户指定顺序):
    0-3:   左颞链       FP1-F7, F7-T3, T3-T5, T5-O1
    4-7:   右颞链       FP2-F8, F8-T4, T4-T6, T6-O2
    8-11:  左副矢状链   FP1-F3, F3-C3, C3-P3, P3-O1
    12-15: 右副矢状链   FP2-F4, F4-C4, C4-P4, P4-O2
    16-21: 中央链       A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2

SOZ标签映射:
    - 单极标注: 直接映射到19通道
    - 双极标注: 通过导联对贡献加权反向映射到19通道

Usage:
    python eeg_pipeline.py --annotation ann.csv --output out.npz
    python eeg_pipeline.py --annotation tusz_manifest.csv --format tusz --data-root F:/dataset/TUSZ/v2.0.3/edf
    python eeg_pipeline.py --annotation bipolar_manifest.csv --format private --data-root E:/DataSet/EEG/EEG_dataset_SUAT
"""

import json
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import mne
mne.set_log_level('ERROR')

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义
# =============================================================================

# 标准19通道 (10-20系统)
STANDARD_19 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]
STD19_IDX = {ch: i for i, ch in enumerate(STANDARD_19)}
N_STD = 19

# TCP所需21电极 (标准19 + A1/A2, 用于中央链计算)
ELECTRODES_21 = STANDARD_19 + ['A1', 'A2']
ELEC21_IDX = {ch: i for i, ch in enumerate(ELECTRODES_21)}
N_ELEC = 21

# TCP 22通道双极导联 — 按用户指定顺序
TCP_PAIRS: List[Tuple[str, str]] = [
    # 左颞链 (0-3)
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    # 右颞链 (4-7)
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    # 左副矢状链 (8-11)
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    # 右副矢状链 (12-15)
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    # 中央链 (16-21)
    ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
    ('C4', 'T4'), ('T4', 'A2'),
]
TCP_NAMES = [f"{a}-{b}" for a, b in TCP_PAIRS]
TCP_IDX = {n: i for i, n in enumerate(TCP_NAMES)}
N_TCP = 22

# 通道名别名
_ALIASES = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    'SPH-R': 'SPHR', 'SPH-L': 'SPHL',
    'SP1': 'SPHL', 'SP2': 'SPHR',
    'SPH1': 'SPHL', 'SPH2': 'SPHR',
}

# =============================================================================
# CHB-MIT 双极通道映射
# =============================================================================

# CHB-MIT 双极通道名 (10-10 命名) → TCP 22 通道索引
_CHB_BIPOLAR_TO_TCP: Dict[str, int] = {
    'FP1-F7': 0,  'F7-T7': 1,  'T7-P7': 2,  'P7-O1': 3,
    'FP2-F8': 4,  'F8-T8': 5,  'T8-P8': 6,  'P8-O2': 7,
    'FP1-F3': 8,  'F3-C3': 9,  'C3-P3': 10, 'P3-O1': 11,
    'FP2-F4': 12, 'F4-C4': 13, 'C4-P4': 14, 'P4-O2': 15,
}

# 可重建通道: TCP索引 → (加法通道列表, 减法通道列表)
# T3-C3 = (FP1-F3 + F3-C3) − (FP1-F7 + F7-T7)
# C4-T4 = (FP2-F8 + F8-T8) − (FP2-F4 + F4-C4)
_CHB_RECONSTRUCT = {
    17: (['FP1-F3', 'F3-C3'], ['FP1-F7', 'F7-T7']),   # T3-C3
    20: (['FP2-F8', 'F8-T8'], ['FP2-F4', 'F4-C4']),   # C4-T4
}

# CHB-MIT 通道掩码 (18/22 有效)
_CHB_CHANNEL_MASK = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     0, 1, 0, 0, 1, 0],
    dtype=np.float32,
)


# =============================================================================
# 配置
# =============================================================================

@dataclass
class PipelineConfig:
    """预处理管道配置"""
    # ---- 采样 ----
    target_fs: float = 200.0

    # ---- 滤波 (Butterworth IIR, 零相位) ----
    filter_low: float = 3.0
    filter_high: float = 45.0
    filter_order: int = 4

    # ---- 幅值裁剪 ----
    clip_n_std: float = 1.0   # ±1 std, 基于基线期计算

    # ---- 窗口 ----
    pre_onset_sec: float = 5.0   # 发作前秒数
    post_onset_sec: float = 5.0  # 发作后秒数
    # total = 10s → 2000 samples @ 200Hz

    # ---- 补丁 ----
    n_patches: int = 10
    patch_len: int = 200   # samples (1.0s @ 200Hz)

    # ---- 坏段检测 ----
    bad_amp_uv: float = 500.0   # μV 阈值
    flat_sec: float = 1.0       # 平坦段秒数阈值

    # ---- 最小数据要求 ----
    min_pre_sec: float = 3.0
    min_post_sec: float = 3.0
    min_valid_channels: int = 18   # 有效双极导联最低数 (22通道中)

    # ---- 路径 ----
    output_root: str = r'F:\process_dataset'
    tusz_data_root: str = r'F:\dataset\TUSZ\v2.0.3\edf'
    private_data_roots: List[str] = field(default_factory=lambda: [
        r'E:\DataSet\EEG\EEG dataset_SUAT',
    ])

    @property
    def window_samples(self) -> int:
        return int((self.pre_onset_sec + self.post_onset_sec) * self.target_fs)

    @property
    def baseline_samples(self) -> int:
        return int(self.pre_onset_sec * self.target_fs)

    @property
    def bad_amp_v(self) -> float:
        return self.bad_amp_uv * 1e-6


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class SeizureEvent:
    """单个发作事件"""
    edf_path: str
    onset: float             # 发作起始时间 (秒)
    end: float               # 发作结束时间 (秒)
    soz_channels: List[str]  # SOZ通道名列表
    soz_type: str = 'unipolar'   # 'unipolar' / 'bipolar'
    source: str = 'public'       # 'public' / 'private'
    patient_id: str = ''


# =============================================================================
# 工具函数
# =============================================================================

def normalize_ch(name: str) -> str:
    """
    标准化通道名:
        "EEG FP1-REF" → "FP1"
        "Fp1"         → "FP1"
        "T7"          → "T3"
    """
    s = name.strip().upper()
    # EDF前缀
    for pfx in ('EEG ', 'EEG-'):
        if s.startswith(pfx):
            s = s[len(pfx):]
            for sfx in ('-REF', '-LE', '-AR', '-AVG'):
                if s.endswith(sfx):
                    s = s[:-len(sfx)]
            break
    return _ALIASES.get(s, s)


def build_bipolar_to_unipolar_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    构建 19×22 映射矩阵: 双极导联 → 19单极通道

    M[i, j] = 1  ⟺  第i个单极电极参与第j个双极对
    participation[i] = 第i个电极参与的双极对总数

    Returns:
        (M, participation)
    """
    M = np.zeros((N_STD, N_TCP), dtype=np.float32)
    for j, (a, b) in enumerate(TCP_PAIRS):
        if a in STD19_IDX:
            M[STD19_IDX[a], j] = 1.0
        if b in STD19_IDX:
            M[STD19_IDX[b], j] = 1.0
    participation = np.maximum(M.sum(axis=1), 1.0)
    return M, participation


def build_adjacency_matrix() -> np.ndarray:
    """构建 22×22 空间邻接矩阵 (共享电极 → 相邻)"""
    adj = np.zeros((N_TCP, N_TCP), dtype=np.float32)
    for i in range(N_TCP):
        for j in range(i + 1, N_TCP):
            if set(TCP_PAIRS[i]) & set(TCP_PAIRS[j]):
                adj[i, j] = adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


# =============================================================================
# 主管道类
# =============================================================================

class EEGPipeline:
    """
    EEG预处理管道

    对每个EDF文件执行: 读取 → 提取通道 → MNE滤波 → MNE重采样 (缓存)
    对每个发作事件: 提取窗口 → 坏段检测 → 裁剪 → 双极转换 → 标准化 → 分补丁
    """

    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._M, self._participation = build_bipolar_to_unipolar_matrix()

    # -----------------------------------------------------------------
    # 文件读取与预处理 (每个文件只执行一次, 结果缓存)
    # -----------------------------------------------------------------

    def load_edf(self, edf_path: str, onset: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        读取EDF → 截取有效段 (若提供onset) → 提取21电极 → MNE滤波 → MNE重采样

        Returns:
            (data_21, fs)  data_21: (21, n_samples) 单位Volts
        """
        # If cache exists and we aren't cropping explicitly (or it's close enough), use it
        # Note: if cropping is used, caching the whole file no longer makes sense per-file unless we cache per-event.
        cache_key = f"{edf_path}_{onset:.1f}" if onset is not None else edf_path
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. 读取
        raw = self._read_raw(edf_path)

        # 1.5 如果提供了onset，提前Crop以节省内存
        if onset is not None:
            pre = self.cfg.pre_onset_sec + 2.0  # extra margin for edge effects
            post = self.cfg.post_onset_sec + 2.0
            tmin = max(0.0, onset - pre)
            tmax = min(raw.times[-1], onset + post)
            if tmax > tmin:
                raw.crop(tmin=tmin, tmax=tmax)

        # 2. 筛选目标通道
        raw = self._pick_channels(raw)

        # 3. MNE带通滤波 (Butterworth 4阶, 零相位)
        nyq = raw.info['sfreq'] / 2.0
        h_freq = min(self.cfg.filter_high, nyq - 1.0)
        if h_freq > self.cfg.filter_low:
            raw.filter(
                l_freq=self.cfg.filter_low,
                h_freq=h_freq,
                method='iir',
                iir_params=dict(
                    order=self.cfg.filter_order,
                    ftype='butter',
                    output='sos',
                ),
                phase='zero',
                verbose=False,
            )

        # 4. MNE重采样
        if abs(raw.info['sfreq'] - self.cfg.target_fs) > 0.1:
            raw.resample(self.cfg.target_fs, verbose=False)

        fs = float(raw.info['sfreq'])
        raw_data = raw.get_data()   # 单位: Volts
        raw_names = raw.ch_names

        # 5. 按ELECTRODES_21顺序组装 (21, n_samples)
        norm_to_row: Dict[str, int] = {}
        for k, name in enumerate(raw_names):
            n = normalize_ch(name)
            if n not in norm_to_row:
                norm_to_row[n] = k

        data_21 = np.zeros((N_ELEC, raw_data.shape[1]), dtype=np.float64)
        for i, target in enumerate(ELECTRODES_21):
            if target in norm_to_row:
                data_21[i] = raw_data[norm_to_row[target]]

        # Store in cache with the specific key
        self._cache[cache_key] = (data_21, fs)
        return data_21, fs

    def _read_raw(self, path: str) -> mne.io.BaseRaw:
        ext = Path(path).suffix.lower()
        if ext == '.set':
            return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        for enc in ('utf-8', 'latin-1'):
            try:
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False, encoding=enc)

                if raw.annotations:
                    for i, (onset, dur, desc) in enumerate(zip(
                            raw.annotations.onset,
                            raw.annotations.duration,
                            raw.annotations.description
                    )):
                        if onset > raw.times[-1] or onset < 0:
                            print(f"File: {path}")
                            print(f"Data duration: {raw.times[-1]:.2f} seconds")
                            print(f"Number of annotations: {len(raw.annotations)}")
                            print(f"  ⚠️  Annotation {i}: onset={onset:.2f}s, duration={dur:.2f}s, desc={desc}")
                return raw
            except Exception:
                continue
        raise RuntimeError(f"无法读取: {path}")

    def _pick_channels(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """只保留与ELECTRODES_21匹配的通道, 减少后续计算量"""
        target_set = set(ELECTRODES_21)
        seen = set()
        to_pick = []
        for name in raw.ch_names:
            n = normalize_ch(name)
            if n in target_set and n not in seen:
                to_pick.append(name)
                seen.add(n)
        if to_pick:
            raw.pick_channels(to_pick)
        else:
            logger.warning(f"无匹配通道, 将保留全部: {raw.ch_names[:5]}...")
        return raw

    # -----------------------------------------------------------------
    # 窗口提取
    # -----------------------------------------------------------------

    def extract_window(
        self, data: np.ndarray, fs: float, onset: float,
    ) -> Optional[np.ndarray]:
        """
        提取发作窗口 [onset - pre, onset + post]

        Returns:
            (21, window_samples)  或 None (数据不足)
        """
        pre = int(self.cfg.pre_onset_sec * fs)
        post = int(self.cfg.post_onset_sec * fs)
        min_pre = int(self.cfg.min_pre_sec * fs)
        min_post = int(self.cfg.min_post_sec * fs)
        win_len = pre + post
        n_total = data.shape[1]
        onset_samp_abs = int(onset * fs)

        def _calc_indices(onset_samp_ref: int) -> Optional[Tuple[int, int, int, int]]:
            avail_pre = min(pre, max(0, onset_samp_ref))
            avail_post = min(post, max(0, n_total - onset_samp_ref))
            if avail_pre < min_pre or avail_post < min_post:
                return None

            src_s = max(0, onset_samp_ref - pre)
            src_e = min(n_total, onset_samp_ref + post)
            dst_s = pre - (onset_samp_ref - src_s)
            dst_e = dst_s + (src_e - src_s)
            return src_s, src_e, dst_s, dst_e

        indices = _calc_indices(onset_samp_abs)
        if indices is None:
            crop_margin_sec = 2.0
            cropped_tmin_sec = max(0.0, onset - (self.cfg.pre_onset_sec + crop_margin_sec))
            onset_samp_rel = int((onset - cropped_tmin_sec) * fs)
            indices = _calc_indices(onset_samp_rel)
            if indices is None:
                return None

        src_s, src_e, dst_s, dst_e = indices
        window = np.zeros((data.shape[0], win_len), dtype=np.float64)
        window[:, dst_s:dst_e] = data[:, src_s:src_e]
        return window

    # -----------------------------------------------------------------
    # 坏段检测
    # -----------------------------------------------------------------

    def _is_bad_window_legacy(self, window: np.ndarray, fs: float) -> bool:
        """
        检测坏窗口:
          - 幅值 > 500μV (对MNE返回的Volts数据: > 5e-4 V)
          - 平坦段 > 1秒 (连续相同采样 > fs 个点)
        """
        # 幅值检查
        if np.any(np.abs(window) > self.cfg.bad_amp_v):
            return True

        # 平坦段检查
        flat_limit = int(self.cfg.flat_sec * fs)
        for ch in range(window.shape[0]):
            if self._max_flat_run(window[ch]) >= flat_limit:
                return True
        return False

    def is_bad_bipolar_window(
        self,
        window: np.ndarray,
        channel_mask: np.ndarray,
        fs: float,
    ) -> bool:
        """
        DeepSOZ-like permissive artifact screening.

        The check runs after bipolar conversion and only rejects windows that
        are clearly unusable, instead of dropping mildly noisy onset windows.
        """
        valid = np.asarray(channel_mask, dtype=np.float32) > 0.5
        n_valid = int(valid.sum())
        if n_valid <= 0:
            return True

        valid_window = np.asarray(window[valid], dtype=np.float64)
        if valid_window.size == 0:
            return True

        active_cols = np.any(np.abs(valid_window) > 1e-12, axis=0)
        if np.any(active_cols):
            active_idx = np.flatnonzero(active_cols)
            valid_window = valid_window[:, active_idx[0]: active_idx[-1] + 1]
        if valid_window.shape[1] <= 1:
            return True
        if not np.isfinite(valid_window).all():
            return True

        bad_amp_uv = max(float(getattr(self.cfg, 'bad_amp_uv', 1000.0)), 2000.0)
        bad_amp_v = bad_amp_uv * 1e-6
        bad_amp_sample_ratio = max(float(getattr(self.cfg, 'bad_amp_sample_ratio', 0.01)), 0.05)
        bad_channel_ratio = max(float(getattr(self.cfg, 'bad_channel_ratio', 0.25)), 0.50)
        flat_sec = max(float(getattr(self.cfg, 'flat_sec', 2.0)), 4.0)
        flat_channel_ratio = max(float(getattr(self.cfg, 'flat_channel_ratio', 0.25)), 0.50)

        amp_ratio = np.mean(np.abs(valid_window) > bad_amp_v, axis=1)
        n_amp_bad = int(np.sum(amp_ratio >= bad_amp_sample_ratio))
        if (n_amp_bad / float(n_valid)) >= bad_channel_ratio:
            return True

        flat_limit = max(int(flat_sec * fs), 1)
        n_flat_bad = 0
        for ch in valid_window:
            if self._max_flat_run(ch) >= flat_limit:
                n_flat_bad += 1
        if (n_flat_bad / float(n_valid)) >= flat_channel_ratio:
            return True

        return False

    def is_bad_window(self, window: np.ndarray, fs: float) -> bool:
        """
        Backward-compatible raw-window check with DeepSOZ-like permissive
        rejection. This is mainly kept for non-stage code paths.
        """
        valid = np.any(np.abs(window) > 1e-12, axis=1)
        n_valid = int(valid.sum())
        if n_valid <= 0:
            return True

        valid_window = np.asarray(window[valid], dtype=np.float64)
        active_cols = np.any(np.abs(valid_window) > 1e-12, axis=0)
        if np.any(active_cols):
            active_idx = np.flatnonzero(active_cols)
            valid_window = valid_window[:, active_idx[0]: active_idx[-1] + 1]
        if valid_window.shape[1] <= 1:
            return True
        if not np.isfinite(valid_window).all():
            return True

        bad_amp_uv = max(float(getattr(self.cfg, 'bad_amp_uv', 1000.0)), 2000.0)
        bad_amp_v = bad_amp_uv * 1e-6
        bad_amp_sample_ratio = max(float(getattr(self.cfg, 'bad_amp_sample_ratio', 0.01)), 0.05)
        bad_channel_ratio = max(float(getattr(self.cfg, 'bad_channel_ratio', 0.25)), 0.50)
        flat_sec = max(float(getattr(self.cfg, 'flat_sec', 2.0)), 4.0)
        flat_channel_ratio = max(float(getattr(self.cfg, 'flat_channel_ratio', 0.25)), 0.50)

        amp_ratio = np.mean(np.abs(valid_window) > bad_amp_v, axis=1)
        n_amp_bad = int(np.sum(amp_ratio >= bad_amp_sample_ratio))
        if (n_amp_bad / float(n_valid)) >= bad_channel_ratio:
            return True

        flat_limit = max(int(flat_sec * fs), 1)
        n_flat_bad = 0
        for ch in valid_window:
            if self._max_flat_run(ch) >= flat_limit:
                n_flat_bad += 1
        if (n_flat_bad / float(n_valid)) >= flat_channel_ratio:
            return True

        return False

    @staticmethod
    def _max_flat_run(ch: np.ndarray) -> int:
        """最长连续近零差分段长度"""
        if len(ch) < 2:
            return 0
        diff = np.abs(np.diff(ch))
        is_flat = diff < 1e-10
        if not np.any(is_flat):
            return 0
        padded = np.concatenate(([False], is_flat, [False]))
        d = np.diff(padded.astype(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        if len(starts) == 0:
            return 0
        return int(np.max(ends - starts))

    # -----------------------------------------------------------------
    # 幅值裁剪 (基于基线期)
    # -----------------------------------------------------------------

    def clip_by_baseline(
        self, window: np.ndarray, baseline_n: int,
    ) -> np.ndarray:
        """
        幅值裁剪: 每通道 clip 到 [baseline_mean ± clip_n_std * baseline_std]

        Args:
            window:     (n_ch, n_samples) 完整窗口
            baseline_n: 基线采样点数 (发作前部分)
        """
        bl = window[:, :baseline_n]
        mu = np.mean(bl, axis=1, keepdims=True)
        sd = np.std(bl, axis=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        lo = mu - self.cfg.clip_n_std * sd
        hi = mu + self.cfg.clip_n_std * sd
        return np.clip(window, lo, hi)

    # -----------------------------------------------------------------
    # TCP双极转换
    # -----------------------------------------------------------------

    def to_tcp_bipolar(
        self, data_21: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        21单极 → 22 TCP双极  (bipolar[i] = anode - cathode)

        Returns:
            (bipolar, mask)
            bipolar: (22, n_samples)
            mask:    (22,)  1=有效, 0=缺失电极
        """
        n_samp = data_21.shape[1]
        bipolar = np.zeros((N_TCP, n_samp), dtype=np.float64)
        mask = np.zeros(N_TCP, dtype=np.float32)

        for i, (anode, cathode) in enumerate(TCP_PAIRS):
            a_i = ELEC21_IDX.get(anode)
            c_i = ELEC21_IDX.get(cathode)
            if a_i is None or c_i is None:
                continue
            a_ok = np.any(data_21[a_i] != 0)
            c_ok = np.any(data_21[c_i] != 0)
            if a_ok and c_ok:
                bipolar[i] = data_21[a_i] - data_21[c_i]
                mask[i] = 1.0

        return bipolar, mask

    # -----------------------------------------------------------------
    # 标准化
    # -----------------------------------------------------------------

    @staticmethod
    def normalize_by_baseline(
        data: np.ndarray, baseline_n: int,
    ) -> np.ndarray:
        """
        基于基线期的 z-score 标准化

        Args:
            data:       (n_ch, n_samples) 双极数据
            baseline_n: 基线采样点数
        """
        bl = data[:, :baseline_n]
        mu = np.mean(bl, axis=1, keepdims=True)
        sd = np.std(bl, axis=1, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return (data - mu) / sd

    # -----------------------------------------------------------------
    # 补丁分割
    # -----------------------------------------------------------------

    def to_patches(self, data: np.ndarray) -> np.ndarray:
        """
        (22, 2000) → (22, 20, 100)

        若样本数不足, 零填充; 超出则截断。
        """
        n_ch, n_samp = data.shape
        expected = self.cfg.n_patches * self.cfg.patch_len  # 2000

        if n_samp < expected:
            padded = np.zeros((n_ch, expected), dtype=data.dtype)
            padded[:, :n_samp] = data
            data = padded
        elif n_samp > expected:
            data = data[:, :expected]

        return data.reshape(n_ch, self.cfg.n_patches, self.cfg.patch_len)

    # -----------------------------------------------------------------
    # SOZ标签映射
    # -----------------------------------------------------------------

    def map_soz_to_19ch(
        self,
        soz_channels: List[str],
        soz_type: str = 'unipolar',
    ) -> np.ndarray:
        """
        SOZ标注 → 19通道二值标签

        单极标注: 直接映射
        双极标注: 导联对贡献加权反向映射
            - 若TCP(A-B)为SOZ, 则A和B各获得 1/N 贡献 (N=该电极参与的导联对数)
            - 贡献 > 0 → 标记为SOZ

        Returns:
            (19,) float32 二值标签
        """
        labels = np.zeros(N_STD, dtype=np.float32)

        if soz_type == 'unipolar':
            for ch in soz_channels:
                n = normalize_ch(ch)
                if n in STD19_IDX:
                    labels[STD19_IDX[n]] = 1.0

        elif soz_type == 'bipolar':
            bipolar_soz = np.zeros(N_TCP, dtype=np.float32)
            for ch in soz_channels:
                n = ch.strip().upper()
                if n in TCP_IDX:
                    bipolar_soz[TCP_IDX[n]] = 1.0

            # 加权反向映射:  score[i] = Σ M[i,j]*soz[j] / participation[i]
            raw_scores = self._M @ bipolar_soz
            weighted = raw_scores / self._participation
            labels = (weighted > 0).astype(np.float32)

        return labels

    # -----------------------------------------------------------------
    # 单事件处理
    # -----------------------------------------------------------------

    def process_event(self, event: SeizureEvent) -> Optional[Dict]:
        """
        处理单个发作事件, 完整流程:
            load → window → bad_check → clip → bipolar → normalize → patches

        Returns:
            {'X': (22,20,100), 'y_soz': (19,), 'channel_mask': (22,),
             'metadata': dict}
            或 None (被剔除)
        """
        # (a) 读取 + 滤波 + 重采样 (缓存)
        try:
            data_21, fs = self.load_edf(event.edf_path, onset=event.onset)
        except Exception as e:
            logger.error(f"读取失败 {event.edf_path}: {e}")
            return None

        bl_n = int(self.cfg.pre_onset_sec * fs)

        # (d) 提取窗口 [onset-5s, onset+5s]
        window = self.extract_window(data_21, fs, event.onset)
        if window is None:
            logger.debug(f"数据不足: onset={event.onset:.1f}s")
            return None

        # (d) 坏段检测 — 在裁剪前检查原始幅值
        if False and self.is_bad_window(window, fs):
            logger.debug(f"坏段剔除: onset={event.onset:.1f}s")
            return None

        # (b/e) 幅值裁剪 (基于基线期)
        window = self.clip_by_baseline(window, bl_n)

        # (e) TCP双极转换: 21ch → 22ch
        bipolar, ch_mask = self.to_tcp_bipolar(window)

        # 有效导联数检查
        n_valid = int(ch_mask.sum())
        if n_valid < self.cfg.min_valid_channels:
            logger.debug(
                f"有效导联不足: {n_valid}/{N_TCP} < {self.cfg.min_valid_channels}, "
                f"edf={event.edf_path}"
            )
            return None

        # (f) 标准化: 基于双极基线期 z-score
        if self.is_bad_bipolar_window(bipolar, ch_mask, fs):
            logger.debug(f"é§å¿”î†Œé“æ—ˆæ«Ž: onset={event.onset:.1f}s")
            return None

        bipolar = self.normalize_by_baseline(bipolar, bl_n)

        # (g) 补丁分割: (22, 2000) → (22, 20, 100)
        patches = self.to_patches(bipolar)

        # SOZ标签
        y_soz = self.map_soz_to_19ch(event.soz_channels, event.soz_type)

        return {
            'X': patches.astype(np.float32),
            'y_soz': y_soz,
            'channel_mask': ch_mask,
            'metadata': {
                'source': event.source,
                'patient_id': event.patient_id,
                'edf_path': event.edf_path,
                'onset': event.onset,
            },
        }

    # -----------------------------------------------------------------
    # CHB-MIT 双极 EDF 处理
    # -----------------------------------------------------------------

    def load_bipolar_edf(
        self, edf_path: str, onset: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        读取 CHB-MIT 双极 EDF → 映射到 TCP 22 通道 + 重建 T3-C3/C4-T4

        CHB-MIT 的 EDF 文件存储预计算的双极差分信号，
        使用 10-10 命名 (T7/T8/P7/P8)。

        Returns:
            (bipolar_22, channel_mask, fs)
            bipolar_22: (22, n_samples) 单位 Volts
            channel_mask: (22,) 有效通道掩码 (18/22)
            fs: 采样率 (Hz)
        """
        # 1. 读取 EDF
        raw = self._read_raw(edf_path)

        # 1.5 Crop 以节省内存
        if onset is not None:
            pre = self.cfg.pre_onset_sec + 2.0
            post = self.cfg.post_onset_sec + 2.0
            tmin = max(0.0, onset - pre)
            tmax = min(raw.times[-1], onset + post)
            if tmax > tmin:
                raw.crop(tmin=tmin, tmax=tmax)

        # 2. 滤波
        nyq = raw.info['sfreq'] / 2.0
        h_freq = min(self.cfg.filter_high, nyq - 1.0)
        if h_freq > self.cfg.filter_low:
            raw.filter(
                l_freq=self.cfg.filter_low,
                h_freq=h_freq,
                method='iir',
                iir_params=dict(
                    order=self.cfg.filter_order,
                    ftype='butter',
                    output='sos',
                ),
                phase='zero',
                verbose=False,
            )

        # 3. 重采样
        if abs(raw.info['sfreq'] - self.cfg.target_fs) > 0.1:
            raw.resample(self.cfg.target_fs, verbose=False)

        fs = float(raw.info['sfreq'])
        raw_data = raw.get_data()  # (n_channels, n_samples)
        raw_names = raw.ch_names

        # 4. 建立通道名映射 (大写 + 去除空格/前缀)
        ch_name_to_idx: Dict[str, int] = {}
        for k, name in enumerate(raw_names):
            # 标准化: 去空格, 大写
            n = name.strip().upper()
            # 去除常见前缀 (如 "EEG ")
            for pfx in ('EEG ', 'EEG-'):
                if n.startswith(pfx):
                    n = n[len(pfx):]
                    break
            if n not in ch_name_to_idx:
                ch_name_to_idx[n] = k

        n_samp = raw_data.shape[1]
        bipolar_22 = np.zeros((N_TCP, n_samp), dtype=np.float64)
        ch_mask = np.array(_CHB_CHANNEL_MASK, dtype=np.float32)

        # 5. 直接映射 16 个通道
        for chb_name, tcp_idx in _CHB_BIPOLAR_TO_TCP.items():
            if chb_name in ch_name_to_idx:
                bipolar_22[tcp_idx] = raw_data[ch_name_to_idx[chb_name]]
            else:
                ch_mask[tcp_idx] = 0.0

        # 6. 重建 T3-C3 (idx=17) 和 C4-T4 (idx=20)
        for tcp_idx, (add_chs, sub_chs) in _CHB_RECONSTRUCT.items():
            add_ok = all(ch in ch_name_to_idx for ch in add_chs)
            sub_ok = all(ch in ch_name_to_idx for ch in sub_chs)
            if add_ok and sub_ok:
                signal = np.zeros(n_samp, dtype=np.float64)
                for ch in add_chs:
                    signal += raw_data[ch_name_to_idx[ch]]
                for ch in sub_chs:
                    signal -= raw_data[ch_name_to_idx[ch]]
                bipolar_22[tcp_idx] = signal
                ch_mask[tcp_idx] = 1.0
            else:
                ch_mask[tcp_idx] = 0.0

        return bipolar_22, ch_mask, fs

    def process_event_bipolar(self, event: SeizureEvent) -> Optional[Dict]:
        """
        处理 CHB-MIT 等双极 EDF 数据集的发作事件。

        与 process_event() 流程类似，但跳过单极→双极转换步骤，
        直接从双极 EDF 读取数据。

        Returns:
            {'X': (22, n_patches, patch_len), 'y_soz': (19,),
             'channel_mask': (22,), 'metadata': dict}
            或 None (被剔除)
        """
        try:
            bipolar, ch_mask, fs = self.load_bipolar_edf(
                event.edf_path, onset=event.onset,
            )
        except Exception as e:
            logger.error(f"读取双极 EDF 失败 {event.edf_path}: {e}")
            return None

        bl_n = int(self.cfg.pre_onset_sec * fs)

        # 提取窗口 [onset-pre, onset+post]
        n_samp = bipolar.shape[1]
        pre = int(self.cfg.pre_onset_sec * fs)
        post = int(self.cfg.post_onset_sec * fs)
        win_len = pre + post

        # 由于已经 crop 过, onset 需要转换为相对时间
        crop_margin = self.cfg.pre_onset_sec + 2.0
        onset_abs = event.onset
        cropped_tmin = max(0.0, onset_abs - crop_margin)
        onset_rel = onset_abs - cropped_tmin
        onset_samp = int(onset_rel * fs)

        avail_pre = min(pre, max(0, onset_samp))
        avail_post = min(post, max(0, n_samp - onset_samp))
        min_pre = int(self.cfg.min_pre_sec * fs)
        min_post = int(self.cfg.min_post_sec * fs)

        if avail_pre < min_pre or avail_post < min_post:
            logger.debug(f"数据不足: onset={event.onset:.1f}s")
            return None

        src_s = max(0, onset_samp - pre)
        src_e = min(n_samp, onset_samp + post)
        dst_s = pre - (onset_samp - src_s)
        dst_e = dst_s + (src_e - src_s)

        window = np.zeros((N_TCP, win_len), dtype=np.float64)
        window[:, dst_s:dst_e] = bipolar[:, src_s:src_e]

        # 幅值裁剪
        window = self.clip_by_baseline(window, bl_n)

        # 有效导联数检查 (CHB-MIT 默认 18 有效)
        n_valid = int(ch_mask.sum())
        if n_valid < self.cfg.min_valid_channels:
            logger.debug(
                f"有效导联不足: {n_valid}/{N_TCP} < {self.cfg.min_valid_channels}, "
                f"edf={event.edf_path}"
            )
            return None

        # 坏段检测
        if self.is_bad_bipolar_window(window, ch_mask, fs):
            logger.debug(f"坏段剔除 (bipolar): onset={event.onset:.1f}s")
            return None

        # 标准化
        window = self.normalize_by_baseline(window, bl_n)

        # 补丁分割
        patches = self.to_patches(window)

        # SOZ 标签 (CHB-MIT 无 SOZ → 全 0)
        y_soz = self.map_soz_to_19ch(event.soz_channels, event.soz_type)

        return {
            'X': patches.astype(np.float32),
            'y_soz': y_soz,
            'channel_mask': ch_mask,
            'metadata': {
                'source': event.source,
                'patient_id': event.patient_id,
                'edf_path': event.edf_path,
                'onset': event.onset,
            },
        }

    # -----------------------------------------------------------------
    # 批量处理 (返回列表, 不保存)
    # -----------------------------------------------------------------

    def process_all(
        self,
        events: List[SeizureEvent],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        批量处理所有发作事件 (全部保留在内存中)

        Returns:
            X:        (N, 22, 20, 100)   float32
            y_soz:    (N, 19)            float32
            metadata: list[dict]         len=N
        """
        from tqdm import tqdm

        X_all, y_all, meta_all = [], [], []

        iterator = tqdm(events, desc="Preprocessing") if verbose else events
        prev_file = None

        for event in iterator:
            cache_key = f"{event.edf_path}_{event.onset:.1f}"
            if cache_key != prev_file:
                if prev_file is not None and prev_file in self._cache:
                    del self._cache[prev_file]
                prev_file = cache_key

            result = self.process_event(event)
            if result is not None:
                X_all.append(result['X'])
                y_all.append(result['y_soz'])
                meta_all.append(result['metadata'])

        if not X_all:
            logger.warning("No valid samples produced!")
            return (
                np.empty((0, N_TCP, self.cfg.n_patches, self.cfg.patch_len), dtype=np.float32),
                np.empty((0, N_STD), dtype=np.float32),
                [],
            )

        X = np.stack(X_all)
        y_soz = np.stack(y_all)
        return X, y_soz, meta_all

    # -----------------------------------------------------------------
    # 批量处理 + 逐样本保存到磁盘
    # -----------------------------------------------------------------

    def process_and_save(
        self,
        events: List[SeizureEvent],
        output_root: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        处理所有事件, 逐样本保存为 .npz 文件到磁盘

        目录结构:
            {output_root}/
                tusz/                    ← 公共数据
                    {patient_id}/
                        {patient_id}_evt{idx}.npz
                        ...
                private/                 ← 私有数据
                    {patient_id}/
                        {patient_id}_evt{idx}.npz
                        ...
                meta/                    ← 全局元数据
                    adj.npy              ← 22×22 邻接矩阵
                    tcp_channels.json    ← TCP通道名列表
                    std_channels.json    ← 19标准通道名列表
                    config.json          ← 预处理配置
                index_tusz.csv           ← TUSZ样本索引
                index_private.csv        ← 私有样本索引
                index_all.csv            ← 全部样本索引

        每个 .npz 包含:
            X:            (22, 20, 100)  float32
            y_soz:        (19,)          float32
            channel_mask: (22,)          float32
            onset:        scalar         float64
            patient_id:   str
            source:       str

        Returns:
            pd.DataFrame  所有样本的索引信息
        """
        from tqdm import tqdm

        root = Path(output_root or self.cfg.output_root)
        tusz_dir = root / 'tusz'
        private_dir = root / 'private'
        meta_dir = root / 'meta'

        # 创建目录
        for d in [tusz_dir, private_dir, meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 保存全局元数据 (只需一次)
        self._save_global_meta(meta_dir)

        # 统计计数器 (per patient)
        patient_counters: Dict[str, int] = {}
        index_rows: List[Dict] = []

        iterator = tqdm(events, desc="Processing & Saving") if verbose else events
        prev_file = None
        n_total = 0
        n_bad = 0
        n_saved = 0

        for event in iterator:
            n_total += 1

            # 缓存管理
            cache_key = f"{event.edf_path}_{event.onset:.1f}"
            if cache_key != prev_file:
                if prev_file is not None and prev_file in self._cache:
                    del self._cache[prev_file]
                prev_file = cache_key

            result = self.process_event(event)
            if result is None:
                n_bad += 1
                continue

            # 确定保存路径
            pid = result['metadata']['patient_id'] or 'unknown'
            source = result['metadata']['source']
            base_dir = tusz_dir if source == 'public' else private_dir
            patient_dir = base_dir / pid
            patient_dir.mkdir(parents=True, exist_ok=True)

            # 样本编号
            key = f"{source}_{pid}"
            idx = patient_counters.get(key, 0)
            patient_counters[key] = idx + 1

            # 保存单个样本
            npz_name = f"{pid}_evt{idx:04d}.npz"
            npz_path = patient_dir / npz_name
            np.savez_compressed(
                str(npz_path),
                X=result['X'],                     # (22, 20, 100)
                y_soz=result['y_soz'],             # (19,)
                channel_mask=result['channel_mask'],  # (22,)
            )

            # 索引记录
            y_soz = result['y_soz']
            index_rows.append({
                'npz_path': str(npz_path),
                'npz_rel': str(npz_path.relative_to(root)),
                'source': source,
                'patient_id': pid,
                'onset': result['metadata']['onset'],
                'edf_path': result['metadata']['edf_path'],
                'has_soz': int(np.any(y_soz > 0)),
                'n_soz_channels': int(np.sum(y_soz > 0)),
                'soz_channels': ','.join(
                    STANDARD_19[i] for i in range(N_STD) if y_soz[i] > 0
                ),
            })
            n_saved += 1

        # 构建索引DataFrame
        df_index = pd.DataFrame(index_rows)

        # 保存索引CSV
        if not df_index.empty:
            # 全量索引
            idx_all_path = root / 'index_all.csv'
            df_index.to_csv(str(idx_all_path), index=False)
            logger.info(f"Saved: {idx_all_path} ({len(df_index)} samples)")

            # 按source拆分索引
            for src, subdir_name in [('public', 'tusz'), ('private', 'private')]:
                df_sub = df_index[df_index['source'] == src]
                if not df_sub.empty:
                    idx_path = root / f'index_{subdir_name}.csv'
                    df_sub.to_csv(str(idx_path), index=False)
                    logger.info(f"Saved: {idx_path} ({len(df_sub)} samples)")

        # 打印统计
        n_pub = sum(1 for r in index_rows if r['source'] == 'public')
        n_pri = sum(1 for r in index_rows if r['source'] == 'private')
        n_soz = sum(1 for r in index_rows if r['has_soz'])
        n_patients = len(set(r['patient_id'] for r in index_rows))

        logger.info("=" * 60)
        logger.info(f"Preprocessing complete")
        logger.info(f"  Output root:  {root}")
        logger.info(f"  Total events: {n_total}")
        logger.info(f"  Rejected:     {n_bad} (bad/insufficient)")
        logger.info(f"  Saved:        {n_saved}")
        logger.info(f"    public:     {n_pub}")
        logger.info(f"    private:    {n_pri}")
        logger.info(f"  Patients:     {n_patients}")
        logger.info(f"  SOZ positive: {n_soz}")
        logger.info(f"  Sample shape: X=(22,20,100)  y_soz=(19,)")
        logger.info("=" * 60)

        return df_index

    def _save_global_meta(self, meta_dir: Path):
        """保存全局元数据 (邻接矩阵, 通道名, 配置)"""
        # 邻接矩阵
        adj_path = meta_dir / 'adj.npy'
        if not adj_path.exists():
            adj = build_adjacency_matrix()
            np.save(str(adj_path), adj)
            logger.info(f"Saved adjacency matrix: {adj_path}")

        # TCP通道名
        tcp_path = meta_dir / 'tcp_channels.json'
        if not tcp_path.exists():
            with open(str(tcp_path), 'w') as f:
                json.dump(TCP_NAMES, f, indent=2)

        # 标准19通道名
        std_path = meta_dir / 'std_channels.json'
        if not std_path.exists():
            with open(str(std_path), 'w') as f:
                json.dump(STANDARD_19, f, indent=2)

        # 预处理配置
        cfg_path = meta_dir / 'config.json'
        cfg_dict = {
            'target_fs': self.cfg.target_fs,
            'filter_low': self.cfg.filter_low,
            'filter_high': self.cfg.filter_high,
            'filter_order': self.cfg.filter_order,
            'clip_n_std': self.cfg.clip_n_std,
            'pre_onset_sec': self.cfg.pre_onset_sec,
            'post_onset_sec': self.cfg.post_onset_sec,
            'n_patches': self.cfg.n_patches,
            'patch_len': self.cfg.patch_len,
            'bad_amp_uv': self.cfg.bad_amp_uv,
            'flat_sec': self.cfg.flat_sec,
            'n_tcp_channels': N_TCP,
            'n_std_channels': N_STD,
            'tcp_pairs': TCP_NAMES,
            'std_channels': STANDARD_19,
            'window_samples': self.cfg.window_samples,
            'baseline_samples': self.cfg.baseline_samples,
        }
        with open(str(cfg_path), 'w') as f:
            json.dump(cfg_dict, f, indent=2, ensure_ascii=False)

    def clear_cache(self):
        self._cache.clear()


# =============================================================================
# 标注加载器
# =============================================================================

def load_generic_csv(csv_path: str) -> List[SeizureEvent]:
    """
    加载通用标注CSV

    期望列: edf_path, seizure_onset, soz_channels
    可选列: seizure_end, data_source, patient_id
    """
    df = pd.read_csv(csv_path)
    events = []
    for _, row in df.iterrows():
        soz_str = str(row.get('soz_channels', ''))
        soz_list = [ch.strip() for ch in soz_str.split(',') if ch.strip()]
        is_bipolar = any('-' in ch for ch in soz_list)

        events.append(SeizureEvent(
            edf_path=str(row['edf_path']),
            onset=float(row['seizure_onset']),
            end=float(row.get('seizure_end', row['seizure_onset'] + 30)),
            soz_channels=soz_list,
            soz_type='bipolar' if is_bipolar else 'unipolar',
            source=str(row.get('data_source', 'public')),
            patient_id=str(row.get('patient_id', '')),
        ))
    return events


def load_tusz_manifest(
    manifest_path: str,
    data_root: str = '',
) -> List[SeizureEvent]:
    """
    加载TUSZ manifest → SeizureEvent列表

    manifest列: patient_id, edf_path, has_seizure, sz_starts, sz_ends, onset_channels
    """
    df = pd.read_csv(manifest_path)
    df_sz = df[df['has_seizure'] == True].reset_index(drop=True)

    events = []
    for _, row in df_sz.iterrows():
        edf_rel = str(row['edf_path'])
        edf_full = str(Path(data_root) / edf_rel) if data_root else edf_rel

        starts_s = str(row.get('sz_starts', ''))
        ends_s = str(row.get('sz_ends', ''))
        if not starts_s.strip():
            continue

        try:
            starts = [float(t) for t in starts_s.split(';') if t.strip()]
            ends = [float(t) for t in ends_s.split(';') if t.strip()]
        except ValueError:
            continue

        onset_str = str(row.get('onset_channels', ''))
        soz_list = [ch.strip() for ch in onset_str.split(',') if ch.strip()]
        is_bipolar = any('-' in ch for ch in soz_list)

        for onset, end in zip(starts, ends):
            events.append(SeizureEvent(
                edf_path=edf_full,
                onset=onset,
                end=end,
                soz_channels=soz_list,
                soz_type='bipolar' if is_bipolar else 'unipolar',
                source='public',
                patient_id=str(row['patient_id']),
            ))

    logger.info(f"TUSZ manifest: {len(events)} 发作事件 from {len(df_sz)} 文件")
    return events


def load_private_manifest(
    manifest_path: str,
    data_roots: Optional[List[str]] = None,
) -> List[SeizureEvent]:
    """
    加载私有数据manifest → SeizureEvent列表

    SOZ来源: per-electrode列 (fp1, fp2, ...) 值>0 → SOZ
    """
    df = pd.read_csv(manifest_path)
    elec_cols = [
        'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
        't3', 'c3', 'cz', 'c4', 't4',
        't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2',
    ]

    events = []
    for _, row in df.iterrows():
        # 提取SOZ电极
        soz = []
        for col in elec_cols:
            try:
                if int(float(row.get(col, 0))) > 0:
                    soz.append(normalize_ch(col))
            except (ValueError, TypeError):
                pass

        # 查找文件
        loc = str(row.get('loc', ''))
        edf_path = _find_private_file(loc, data_roots or [])
        if not edf_path:
            continue

        try:
            onset = float(row['sz_start'])
            end = float(row['sz_end'])
        except (KeyError, ValueError, TypeError):
            continue

        events.append(SeizureEvent(
            edf_path=edf_path,
            onset=onset,
            end=end,
            soz_channels=soz,
            soz_type='unipolar',
            source='private',
            patient_id=str(row.get('pt_id', '')),
        ))

    logger.info(f"Private manifest: {len(events)} 发作事件")
    return events


def _find_private_file(loc: str, roots: List[str]) -> Optional[str]:
    """查找私有数据文件 (.set格式特殊路径)"""
    stem = Path(loc).stem
    parent = Path(loc).parent

    for root in roots:
        root_p = str(root) + '_processed'
        new_fn = f"{stem}_filtered_3_45_postICA_eye.set"
        for cand in [
            Path(root_p) / parent / new_fn,
            Path(root_p) / new_fn,
        ]:
            if cand.exists():
                return str(cand)
        for found in Path(root_p).rglob(new_fn):
            return str(found)
        simple = Path(root_p) / parent / (stem + '.set')
        if simple.exists():
            return str(simple)

    # fallback: 原始路径
    for root in roots:
        full = Path(root) / loc
        if full.exists():
            return str(full)
    return None


# =============================================================================
# 训练时加载 — PyTorch Dataset
# =============================================================================

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TimeFilterDataset:
    """
    从 F:/process_dataset/ 加载预处理数据的 PyTorch Dataset

    目录结构 (由 process_and_save 生成):
        {root}/
            index_all.csv / index_tusz.csv / index_private.csv
            meta/adj.npy, tcp_channels.json, std_channels.json, config.json
            tusz/{patient_id}/{patient_id}_evt{idx}.npz
            private/{patient_id}/{patient_id}_evt{idx}.npz

    每个 npz:
        X:            (22, 20, 100)  float32
        y_soz:        (19,)          float32
        channel_mask: (22,)          float32

    Usage:
        ds = TimeFilterDataset('F:/process_dataset')
        X, y, mask, meta = ds[0]

        # 也可指定子集
        ds = TimeFilterDataset('F:/process_dataset', subset='tusz')

        # PyTorch DataLoader
        loader = ds.create_dataloader(batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        root: str = r'F:\process_dataset',
        subset: str = 'all',
        patient_ids: Optional[List[str]] = None,
        source_filter: Optional[str] = None,
        soz_only: bool = False,
        preload: bool = False,
    ):
        """
        Args:
            root:          预处理数据根目录
            subset:        'all' / 'tusz' / 'private'
            patient_ids:   仅包含这些患者 (用于train/val/test拆分)
            source_filter: 'public' / 'private' (按来源过滤)
            soz_only:      仅保留SOZ阳性样本
            preload:       True=全部预加载到内存 (数据量小时推荐)
        """
        self.root = Path(root)
        self.preload = preload

        # 加载索引
        index_name = f'index_{subset}.csv' if subset != 'all' else 'index_all.csv'
        index_path = self.root / index_name
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        self.df = pd.read_csv(str(index_path))

        # 过滤
        if source_filter:
            self.df = self.df[self.df['source'] == source_filter].reset_index(drop=True)
        if patient_ids is not None:
            self.df = self.df[self.df['patient_id'].isin(patient_ids)].reset_index(drop=True)
        if soz_only:
            self.df = self.df[self.df['has_soz'] == 1].reset_index(drop=True)

        # 加载全局元数据
        meta_dir = self.root / 'meta'
        self.adj = np.load(str(meta_dir / 'adj.npy'))
        with open(str(meta_dir / 'tcp_channels.json')) as f:
            self.tcp_channels = json.load(f)
        with open(str(meta_dir / 'std_channels.json')) as f:
            self.std_channels = json.load(f)
        with open(str(meta_dir / 'config.json')) as f:
            self.config = json.load(f)

        # 预加载
        self._cache: Optional[List] = None
        if preload:
            self._cache = [self._load_sample(i) for i in range(len(self.df))]

        logger.info(
            f"TimeFilterDataset: {len(self.df)} samples from {index_path.name}, "
            f"patients={self.df['patient_id'].nunique()}, "
            f"SOZ+={self.df['has_soz'].sum()}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            return self._cache[idx]
        return self._load_sample(idx)

    def _load_sample(self, idx: int):
        row = self.df.iloc[idx]

        # npz路径: 优先用绝对路径, 否则拼接root
        npz_path = row['npz_path']
        if not Path(npz_path).exists():
            npz_path = str(self.root / row['npz_rel'])

        d = np.load(npz_path)
        X = d['X']                     # (22, 20, 100) float32
        y_soz = d['y_soz']            # (19,) float32
        channel_mask = d['channel_mask']  # (22,) float32

        meta = {
            'source': row['source'],
            'patient_id': row['patient_id'],
            'onset': row['onset'],
            'has_soz': row['has_soz'],
            'npz_path': npz_path,
        }

        if _HAS_TORCH:
            return (
                torch.from_numpy(X),
                torch.from_numpy(y_soz),
                torch.from_numpy(channel_mask),
                meta,
            )
        return X, y_soz, channel_mask, meta

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def get_patient_ids(self) -> List[str]:
        return sorted(self.df['patient_id'].unique().tolist())

    def get_source_counts(self) -> Dict[str, int]:
        return self.df['source'].value_counts().to_dict()

    def split_by_patient(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        按患者划分 train/val/test, 确保同一患者的样本不跨集

        Returns:
            (train_pids, val_pids, test_pids)
        """
        rng = np.random.RandomState(seed)
        pids = sorted(self.df['patient_id'].unique().tolist())
        rng.shuffle(pids)

        n = len(pids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_pids = pids[:n_train]
        val_pids = pids[n_train:n_train + n_val]
        test_pids = pids[n_train + n_val:]

        logger.info(
            f"Patient split: train={len(train_pids)}, "
            f"val={len(val_pids)}, test={len(test_pids)}"
        )
        return train_pids, val_pids, test_pids

    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        """创建 PyTorch DataLoader"""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed")

        def collate_fn(batch):
            Xs, ys, masks, metas = zip(*batch)
            return (
                torch.stack(Xs),       # (B, 22, 20, 100)
                torch.stack(ys),       # (B, 19)
                torch.stack(masks),    # (B, 22)
                list(metas),           # list of dicts
            )

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )


# =============================================================================
# 兼容性: 保留单文件保存/加载 (可选)
# =============================================================================

def save_outputs(
    path: str,
    X: np.ndarray,
    y_soz: np.ndarray,
    metadata: List[Dict],
    adj: Optional[np.ndarray] = None,
):
    """保存为单个npz文件 (适用于数据量较小的场景)"""
    sources = np.array([m['source'] for m in metadata])
    if adj is None:
        adj = build_adjacency_matrix()

    np.savez_compressed(
        path,
        X=X,
        y_soz=y_soz,
        sources=sources,
        adj=adj,
        tcp_channels=np.array(TCP_NAMES),
        std_channels=np.array(STANDARD_19),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    logger.info(f"Saved (single file): {path}  X={X.shape}")


def load_outputs(path: str) -> Dict:
    """加载单文件npz"""
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d['metadata_json']))
    return {
        'X': d['X'],
        'y_soz': d['y_soz'],
        'sources': d['sources'],
        'adj': d['adj'],
        'tcp_channels': list(d['tcp_channels']),
        'std_channels': list(d['std_channels']),
        'metadata': meta,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='EEG Seizure Preprocessing Pipeline (TimeFilter SOZ)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output directory structure (F:/process_dataset/):
    tusz/                     <- public data (per-patient .npz)
        aaaaagei/
            aaaaagei_evt0000.npz
            ...
    private/                  <- private data (per-patient .npz)
        SUAT001/
            SUAT001_evt0000.npz
            ...
    meta/                     <- global metadata
        adj.npy               <- 22x22 adjacency matrix
        tcp_channels.json     <- TCP channel names
        std_channels.json     <- 19 standard channel names
        config.json           <- preprocessing config
    index_all.csv             <- master index
    index_tusz.csv            <- TUSZ index
    index_private.csv         <- private index

Examples:
    # Process TUSZ only
    python eeg_pipeline.py --annotation tusz_manifest.csv --format tusz \\
        --data-root F:/dataset/TUSZ/v2.0.3/edf

    # Process private only
    python eeg_pipeline.py --annotation bipolar_manifest.csv --format private \\
        --data-root "E:/DataSet/EEG/EEG dataset_SUAT"

    # Process both (combined)
    python eeg_pipeline.py --annotation tusz_manifest.csv --format tusz \\
        --data-root F:/dataset/TUSZ/v2.0.3/edf \\
        --private-annotation bipolar_manifest.csv \\
        --private-root "E:/DataSet/EEG/EEG dataset_SUAT"

    # Custom output directory
    python eeg_pipeline.py --annotation tusz_manifest.csv --format tusz \\
        --data-root F:/dataset/TUSZ/v2.0.3/edf --output-root D:/my_preprocessed

Training data loading:
    from eeg_pipeline import TimeFilterDataset
    ds = TimeFilterDataset('F:/process_dataset', subset='all')
    train_pids, val_pids, test_pids = ds.split_by_patient()
    train_ds = TimeFilterDataset('F:/process_dataset', patient_ids=train_pids)
    loader = train_ds.create_dataloader(batch_size=32)
        """,
    )

    p.add_argument('--annotation', required=True, help='annotation file path')
    p.add_argument('--format', default='generic',
                   choices=['generic', 'tusz', 'private'],
                   help='annotation format')
    p.add_argument('--data-root', default='', help='EDF data root directory')
    p.add_argument('--output-root', default=r'F:\process_dataset',
                   help='output root directory (default: F:/process_dataset)')

    # optional: merge private data
    p.add_argument('--private-annotation', default=None,
                   help='private annotation (combine with --format tusz)')
    p.add_argument('--private-root', default=None,
                   help='private data root directory')

    # preprocessing params
    p.add_argument('--target-fs', type=float, default=200.0)
    p.add_argument('--filter-low', type=float, default=3.0)
    p.add_argument('--filter-high', type=float, default=45.0)
    p.add_argument('--clip-std', type=float, default=1.0)
    p.add_argument('--pre-onset', type=float, default=5.0)
    p.add_argument('--post-onset', type=float, default=5.0)
    p.add_argument('--bad-amp', type=float, default=500.0,
                   help='bad segment amplitude threshold (uV)')
    p.add_argument('--flat-sec', type=float, default=1.0,
                   help='bad segment flatness threshold (seconds)')

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    cfg = PipelineConfig(
        target_fs=args.target_fs,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        clip_n_std=args.clip_std,
        pre_onset_sec=args.pre_onset,
        post_onset_sec=args.post_onset,
        bad_amp_uv=args.bad_amp,
        flat_sec=args.flat_sec,
        output_root=args.output_root,
    )
    if args.data_root:
        cfg.tusz_data_root = args.data_root

    # Load annotations
    events: List[SeizureEvent] = []

    if args.format == 'tusz':
        events.extend(load_tusz_manifest(args.annotation, args.data_root))
    elif args.format == 'private':
        roots = [args.data_root] if args.data_root else cfg.private_data_roots
        events.extend(load_private_manifest(args.annotation, roots))
    else:
        events.extend(load_generic_csv(args.annotation))

    # Optional: merge private data
    if args.private_annotation:
        roots = ([args.private_root] if args.private_root
                 else cfg.private_data_roots)
        events.extend(load_private_manifest(args.private_annotation, roots))

    logger.info(f"Total events to process: {len(events)}")

    if not events:
        logger.error("No events found, exiting")
        return

    # Sort by file path (for cache efficiency)
    events.sort(key=lambda e: e.edf_path)

    # Process and save to disk
    pipeline = EEGPipeline(cfg)
    df_index = pipeline.process_and_save(
        events,
        output_root=args.output_root,
        verbose=True,
    )

    if df_index.empty:
        logger.error("No valid samples produced")
        return

    logger.info(
        f"\nTo load in training:\n"
        f"  from eeg_pipeline import TimeFilterDataset\n"
        f"  ds = TimeFilterDataset('{args.output_root}')\n"
        f"  loader = ds.create_dataloader(batch_size=32)\n"
    )


if __name__ == '__main__':
    main()
