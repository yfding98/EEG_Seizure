#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG数据预处理主脚本 — TCP 22通道双极导联

将EDF/SET原始数据按DeepSOZ流程预处理，转换为TCP双极导联npz文件，
用于TimeFilter模型训练SOZ(癫痫起始区)检测。

支持两种数据源:
    1. TUSZ公共数据: EDF文件, 通道命名 "EEG XX-REF" 或 "EEG XX-LE"
    2. 私有数据:     SET/EDF文件, 通道命名 "Fp1", "A1" 等原始名称

两种数据共用预处理流程:
    1. 读取原始数据 (EDF/SET)
    2. 提取标准21电极 (标准19 + A1 + A2)
    3. 带通滤波 (3-45 Hz, 4阶Butterworth零相位)
    4. 幅值裁剪 (±1 std per channel)
    5. 重采样 (→ 200 Hz)
    6. 提取发作段 + 坏段剔除(私有数据)
    7. 转换到TCP 22通道双极导联 (bipolar = anode - cathode)
    8. 基线标准化 (基于非发作段)
    9. 窗口分割 (12s, 50% overlap)
    10. 保存 .npz

两种数据的差异:
    ┌────────────┬──────────────────────┬──────────────────────┐
    │            │ TUSZ (公共)           │ 私有数据              │
    ├────────────┼──────────────────────┼──────────────────────┤
    │ 文件格式    │ .edf                 │ .set (.edf)          │
    │ 通道命名    │ "EEG FP1-REF"        │ "Fp1"                │
    │ A1/A2      │ ✓ (tcp_ar_a除外)     │ ✓ 全部可用            │
    │ 发作注释    │ manifest onset列     │ manifest sz_start/end│
    │ 坏段       │ 无                   │ mask_segments列       │
    │ 基线       │ 自动从非发作段提取    │ base_line列 / 发作前  │
    │ 标签来源    │ onset_channels列     │ per-electrode SOZ列   │
    └────────────┴──────────────────────┴──────────────────────┘

输出 .npz 内容:
    eeg_data:       (22, window_samples)  TCP双极数据 (float32)
    channel_labels: (22,)                 per-channel SOZ标签
    region_labels:  (5,)                  per-region SOZ标签
    channel_mask:   (22,)                 通道有效标记 (1=有效, 0=缺失)
    is_seizure:     bool                  是否为发作窗口
    patient_id:     str
    file_id:        str
    seizure_idx:    int
    window_idx:     int
    data_type:      str ('tusz'/'private')
    split:          str ('train'/'dev'/'eval')

Usage:
    python preprocess.py --data-type tusz
    python preprocess.py --data-type private
    python preprocess.py --data-type both
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    HAS_PYEDFLIB = False

from config import (
    PreprocessConfig,
    TCP_BIPOLAR_PAIRS,
    TCP_CHANNEL_NAMES,
    TCP_IDX,
    TCP_CHAINS,
    N_TCP_CHANNELS,
    TCP_TO_REGION,
    TCP_TO_HEMISPHERE,
    BRAIN_REGIONS,
    REGION_TO_IDX,
    N_REGIONS,
    STANDARD_21_ELECTRODES,
    PRIVATE_ELECTRODE_LABEL_COLUMNS,
    PRIVATE_REGION_COLUMNS,
    PRIVATE_REGION_TO_UNIFIED,
    TUSZ_MANIFEST_CHANNEL_COLUMNS,
    TUSZ_COL_TO_TCP,
    normalize_channel_name,
    build_tcp_adjacency_matrix,
    get_channel_to_region_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. 文件读取
# ==============================================================================

def read_edf_mne(filepath: str, encoding: str = 'utf-8') -> Tuple[np.ndarray, List[str], float]:
    """使用MNE读取EDF文件"""
    if not HAS_MNE:
        raise ImportError("mne未安装，请运行: pip install mne")
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR', encoding=encoding)
    return raw.get_data(), list(raw.ch_names), float(raw.info['sfreq'])


def read_edf_pyedflib(filepath: str) -> Tuple[np.ndarray, List[str], float]:
    """使用pyedflib读取EDF文件"""
    if not HAS_PYEDFLIB:
        raise ImportError("pyedflib未安装")
    f = pyedflib.EdfReader(filepath)
    try:
        n_ch = f.signals_in_file
        ch_names = f.getSignalLabels()
        fs = float(f.getSampleFrequency(0))
        n_samples = f.getNSamples()[0]
        data = np.zeros((n_ch, n_samples))
        for i in range(n_ch):
            data[i] = f.readSignal(i)
        return data, list(ch_names), fs
    finally:
        f._close()


def read_set(filepath: str) -> Tuple[np.ndarray, List[str], float]:
    """读取EEGLAB .set文件"""
    if not HAS_MNE:
        raise ImportError("mne未安装")
    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose='ERROR')
    return raw.get_data(), list(raw.ch_names), float(raw.info['sfreq'])


def read_data_file(filepath: str) -> Tuple[np.ndarray, List[str], float]:
    """
    自动检测格式读取数据文件

    Returns:
        (data, ch_names, fs)
        - data:     (n_channels, n_samples) 原始数据
        - ch_names: 通道名列表
        - fs:       采样率
    """
    ext = Path(filepath).suffix.lower()
    errors = []

    if ext == '.set':
        try:
            return read_set(filepath)
        except Exception as e:
            errors.append(f"SET: {e}")
    elif ext == '.edf':
        if HAS_PYEDFLIB:
            try:
                return read_edf_pyedflib(filepath)
            except Exception as e:
                errors.append(f"pyedflib: {e}")
        if HAS_MNE:
            for enc in ['utf-8', 'latin-1']:
                try:
                    return read_edf_mne(filepath, encoding=enc)
                except Exception as e:
                    errors.append(f"mne({enc}): {e}")
    else:
        errors.append(f"不支持的格式: {ext}")

    raise RuntimeError(f"无法读取 {filepath}: {'; '.join(errors)}")


# ==============================================================================
# 2. 通道提取
# ==============================================================================

def extract_standard_electrodes(
    data: np.ndarray,
    ch_names: List[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    从原始数据中提取标准21电极

    将各种命名变体统一为标准名（FP1, FP2, ...），提取对应通道数据。
    未找到的电极填零。

    Args:
        data:     (n_raw_channels, n_samples)
        ch_names: 原始通道名列表

    Returns:
        (electrode_data, name_to_idx)
        - electrode_data: (21, n_samples) 标准电极数据
        - name_to_idx:    标准化名 → 在electrode_data中的行索引
    """
    # 建立 标准化名 → 原始数据行索引 的映射
    raw_map: Dict[str, int] = {}
    for i, name in enumerate(ch_names):
        norm = normalize_channel_name(name)
        if norm not in raw_map:
            raw_map[norm] = i

    n_samples = data.shape[1] if data.ndim > 1 else len(data)
    n_elec = len(STANDARD_21_ELECTRODES)
    electrode_data = np.zeros((n_elec, n_samples), dtype=np.float64)
    name_to_idx: Dict[str, int] = {}

    found = []
    missing = []
    for j, elec in enumerate(STANDARD_21_ELECTRODES):
        name_to_idx[elec] = j
        if elec in raw_map:
            electrode_data[j] = data[raw_map[elec]]
            found.append(elec)
        else:
            missing.append(elec)

    if missing:
        logger.debug(f"缺失电极: {missing}")

    return electrode_data, name_to_idx


# ==============================================================================
# 3-5. 信号处理
# ==============================================================================

def bandpass_filter(
    data: np.ndarray, fs: float,
    low: float, high: float, order: int = 4
) -> np.ndarray:
    """带通滤波 (Butterworth, 零相位 filtfilt)"""
    nyq = fs / 2.0
    b, a = signal.butter(order, [max(low / nyq, 0.001), min(high / nyq, 0.999)], btype='band')
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    return np.array([signal.filtfilt(b, a, row) for row in data])


def clip_amplitude(data: np.ndarray, n_std: float = 1.0) -> np.ndarray:
    """
    幅值裁剪 — 每个通道独立裁剪到 [mean ± n_std * std]

    Args:
        data:  (n_channels, n_samples)
        n_std: 裁剪的标准差倍数
    """
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        ch = data[i]
        mu = np.mean(ch)
        sd = np.std(ch)
        if sd > 1e-10:
            out[i] = np.clip(ch, mu - n_std * sd, mu + n_std * sd)
        else:
            out[i] = ch
    return out


def resample_data(data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """重采样到目标采样率"""
    if abs(orig_fs - target_fs) < 0.1:
        return data
    ratio = target_fs / orig_fs
    if data.ndim == 1:
        return signal.resample(data, int(len(data) * ratio))
    n_new = int(data.shape[1] * ratio)
    out = np.zeros((data.shape[0], n_new))
    for i in range(data.shape[0]):
        out[i] = signal.resample(data[i], n_new)
    return out


def preprocess_electrode_data(
    data: np.ndarray, fs: float, cfg: PreprocessConfig
) -> Tuple[np.ndarray, float]:
    """
    预处理步骤3-5: 带通滤波 → 幅值裁剪 → 重采样

    Args:
        data: (n_electrodes, n_samples)
        fs:   采样率
        cfg:  配置

    Returns:
        (processed, output_fs)
    """
    data = bandpass_filter(data, fs, cfg.highpass_fc, cfg.lowpass_fc, cfg.filter_order)
    data = clip_amplitude(data, cfg.clip_n_std)
    if abs(fs - cfg.target_fs) > 0.1:
        data = resample_data(data, fs, cfg.target_fs)
        fs = cfg.target_fs
    return data, fs


# ==============================================================================
# 6. TCP双极导联转换
# ==============================================================================

def convert_to_tcp_bipolar(
    electrode_data: np.ndarray,
    name_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将单极电极数据 → TCP 22通道双极导联

    bipolar[i] = electrode[anode] - electrode[cathode]
    缺失电极对 → 该通道填零, channel_mask 标记为0

    Args:
        electrode_data: (21, n_samples) 标准电极数据
        name_to_idx:    标准化名 → 行索引

    Returns:
        (bipolar_data, channel_mask)
        - bipolar_data: (22, n_samples)
        - channel_mask: (22,) 1=有效, 0=缺失
    """
    n_samples = electrode_data.shape[1]
    bipolar = np.zeros((N_TCP_CHANNELS, n_samples), dtype=np.float64)
    mask = np.zeros(N_TCP_CHANNELS, dtype=np.float32)

    for i, (anode, cathode) in enumerate(TCP_BIPOLAR_PAIRS):
        if anode not in name_to_idx or cathode not in name_to_idx:
            continue
        a_idx = name_to_idx[anode]
        c_idx = name_to_idx[cathode]
        a_data = electrode_data[a_idx]
        c_data = electrode_data[c_idx]
        # 双方都有有效数据（非全零）才计算
        if np.any(a_data != 0) and np.any(c_data != 0):
            bipolar[i] = a_data - c_data
            mask[i] = 1.0

    return bipolar, mask


# ==============================================================================
# 7. 坏段处理 (私有数据)
# ==============================================================================

def parse_mask_segments(mask_str) -> List[Tuple[float, float]]:
    """解析mask_segments字段 → [(start, end), ...]"""
    if pd.isna(mask_str) or mask_str == '' or mask_str is None:
        return []
    try:
        segs = json.loads(str(mask_str))
        return [(float(s[0]), float(s[1])) for s in segs
                if isinstance(s, (list, tuple)) and len(s) >= 2]
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def remove_bad_segments(
    data: np.ndarray, fs: float,
    mask_segments: List[Tuple[float, float]],
    seg_start: float = 0.0,
    seg_end: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    移除坏段，拼接剩余好数据

    Args:
        data: (n_ch, n_samples)
        fs:   采样率
        mask_segments: 坏段绝对时间范围
        seg_start/end: 当前数据段在文件中的绝对时间范围

    Returns:
        (cleaned_data, new_duration_sec)
    """
    if not mask_segments:
        return data, data.shape[1] / fs

    if seg_end is None:
        seg_end = seg_start + data.shape[1] / fs

    n_ch, n_samples = data.shape

    # 转为相对采样点
    sample_masks = []
    for a_start, a_end in mask_segments:
        if a_end <= seg_start or a_start >= seg_end:
            continue
        rel_s = max(0.0, a_start - seg_start)
        rel_e = min(seg_end - seg_start, a_end - seg_start)
        if rel_e > rel_s:
            sample_masks.append((max(0, int(rel_s * fs)), min(n_samples, int(rel_e * fs))))

    if not sample_masks:
        return data, data.shape[1] / fs

    # 合并重叠区间
    sample_masks.sort()
    merged = [sample_masks[0]]
    for s, e in sample_masks[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # 拼接好的部分
    parts = []
    prev = 0
    for s, e in merged:
        if prev < s:
            parts.append(data[:, prev:s])
        prev = e
    if prev < n_samples:
        parts.append(data[:, prev:n_samples])

    if not parts:
        return np.zeros((n_ch, 0)), 0.0

    cleaned = np.concatenate(parts, axis=1)
    return cleaned, cleaned.shape[1] / fs


# ==============================================================================
# 8. 标准化
# ==============================================================================

def compute_baseline_stats(
    baseline_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算基线段 per-channel 均值和标准差

    Returns:
        (mean, std) 各 shape (n_ch, 1)
    """
    mu = np.mean(baseline_data, axis=1, keepdims=True)
    sd = np.std(baseline_data, axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def normalize_by_baseline(
    data: np.ndarray, mu: np.ndarray, sd: np.ndarray,
) -> np.ndarray:
    """基于基线统计标准化"""
    if data.ndim == 3:
        return (data - mu[np.newaxis]) / sd[np.newaxis]
    return (data - mu) / sd


def normalize_zscore(data: np.ndarray) -> np.ndarray:
    """Z-score标准化（每通道独立）"""
    mu = np.mean(data, axis=-1, keepdims=True)
    sd = np.std(data, axis=-1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (data - mu) / sd


# ==============================================================================
# 9. 窗口分割
# ==============================================================================

def segment_into_windows(
    data: np.ndarray, fs: float,
    window_len: float, overlap: float = 0.5,
) -> np.ndarray:
    """
    将数据分割为固定长度窗口

    Args:
        data: (n_ch, n_samples)
        fs:   采样率
        window_len: 秒
        overlap:    重叠比例

    Returns:
        (n_windows, n_ch, window_samples)
    """
    win_samples = int(window_len * fs)
    step = int(win_samples * (1 - overlap))
    n_ch, n_total = data.shape

    if n_total < win_samples:
        padded = np.zeros((n_ch, win_samples))
        padded[:, :n_total] = data
        return padded[np.newaxis]

    starts = list(range(0, n_total - win_samples + 1, step))
    if not starts:
        starts = [0]

    windows = np.stack([data[:, s:s + win_samples] for s in starts])
    return windows


# ==============================================================================
# 10. 标签生成
# ==============================================================================

def get_tusz_channel_labels(row: pd.Series) -> np.ndarray:
    """
    从TUSZ manifest → 22通道SOZ标签

    主要依据: onset_channels 列 (逗号分隔的TCP通道名)
    该列包含全部22个TCP通道的信息(含中央链6通道)。
    """
    labels = np.zeros(N_TCP_CHANNELS, dtype=np.float32)
    onset_str = row.get('onset_channels', '')
    if pd.isna(onset_str) or not str(onset_str).strip():
        return labels

    onset_set = set(ch.strip().upper() for ch in str(onset_str).split(',') if ch.strip())
    for i, ch_name in enumerate(TCP_CHANNEL_NAMES):
        if ch_name.upper() in onset_set:
            labels[i] = 1.0

    return labels


def get_tusz_region_labels(row: pd.Series) -> np.ndarray:
    """从TUSZ manifest → 5脑区SOZ标签"""
    labels = np.zeros(N_REGIONS, dtype=np.float32)
    regions_str = row.get('onset_regions', '')
    if pd.isna(regions_str) or not str(regions_str).strip():
        return labels

    region_set = set(r.strip().lower() for r in str(regions_str).split(',') if r.strip())
    for i, region in enumerate(BRAIN_REGIONS):
        if region in region_set:
            labels[i] = 1.0
    return labels


def get_private_channel_labels(row: pd.Series) -> np.ndarray:
    """
    从私有数据manifest → 22通道TCP SOZ标签

    策略: TCP通道(A,B)为SOZ ⟺ 电极A或电极B在manifest中被标记为SOZ
    """
    labels = np.zeros(N_TCP_CHANNELS, dtype=np.float32)

    # 读取per-electrode SOZ状态
    electrode_soz: Dict[str, int] = {}
    for col in PRIVATE_ELECTRODE_LABEL_COLUMNS:
        val = row.get(col, 0)
        try:
            is_soz = int(float(val))
        except (ValueError, TypeError):
            is_soz = 0
        electrode_soz[normalize_channel_name(col)] = is_soz

    for i, (anode, cathode) in enumerate(TCP_BIPOLAR_PAIRS):
        a_soz = electrode_soz.get(anode, 0)
        c_soz = electrode_soz.get(cathode, 0)
        if a_soz or c_soz:
            labels[i] = 1.0

    return labels


def get_private_region_labels(row: pd.Series) -> np.ndarray:
    """
    从私有数据manifest → 5脑区SOZ标签

    lateralized列直接映射 + 从TCP通道标签推导central/occipital
    """
    labels = np.zeros(N_REGIONS, dtype=np.float32)

    # lateralized列 → unified
    for col in PRIVATE_REGION_COLUMNS:
        val = row.get(col, 0)
        try:
            active = int(float(val))
        except (ValueError, TypeError):
            active = 0
        if active:
            unified = PRIVATE_REGION_TO_UNIFIED.get(col)
            if unified and unified in REGION_TO_IDX:
                labels[REGION_TO_IDX[unified]] = 1.0

    # 补充: central/occipital 从TCP通道标签推导
    ch_labels = get_private_channel_labels(row)
    derived = get_channel_to_region_labels(ch_labels)
    for region in ['central', 'occipital']:
        idx = REGION_TO_IDX[region]
        if derived[idx] > 0:
            labels[idx] = 1.0

    return labels


# ==============================================================================
# 基线提取
# ==============================================================================

def extract_tusz_baseline(
    data: np.ndarray, fs: float,
    sz_starts: List[float], sz_ends: List[float],
    current_sz_idx: int, baseline_duration: float,
) -> Optional[np.ndarray]:
    """
    从TUSZ文件提取基线段（非发作段）

    策略:
    1. 列出所有非发作间隔
    2. 优先选当前发作之前的最长间隔
    3. 否则取任意最长间隔
    4. 至少5s才有效
    """
    file_dur = data.shape[1] / fs
    intervals = sorted(zip(sz_starts, sz_ends))

    # 寻找非发作间隔
    gaps = []
    prev_end = 0
    for s, e in intervals:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < file_dur:
        gaps.append((prev_end, file_dur))

    if not gaps:
        return None

    # 优先选当前发作之前的间隔
    current_start = sz_starts[current_sz_idx]
    best_gap, best_dur = None, 0

    for g_s, g_e in gaps:
        if g_e <= current_start and (g_e - g_s) > best_dur:
            best_dur = g_e - g_s
            best_gap = (g_s, g_e)

    if best_gap is None:
        for g_s, g_e in gaps:
            if (g_e - g_s) > best_dur:
                best_dur = g_e - g_s
                best_gap = (g_s, g_e)

    if best_gap is None or best_dur < 5.0:
        return None

    g_s, g_e = best_gap
    actual_end = min(g_e, g_s + baseline_duration)
    return data[:, int(g_s * fs):int(actual_end * fs)]


def extract_private_baseline(
    data: np.ndarray, fs: float,
    row: pd.Series, baseline_duration: float,
) -> Optional[np.ndarray]:
    """
    从私有数据提取基线段

    优先: manifest中 base_line 列
    Fallback: 发作前数据
    """
    # 尝试manifest中的base_line字段
    bl_str = row.get('base_line', None)
    if bl_str and pd.notna(bl_str) and str(bl_str).strip():
        try:
            bs = str(bl_str).strip()
            if ',' in bs:
                parts = bs.split(',')
                b_start, b_end = float(parts[0]), float(parts[1])
            elif '-' in bs and not bs.startswith('-'):
                parts = bs.split('-')
                b_start, b_end = float(parts[0]), float(parts[1])
            else:
                b_start = float(bs)
                b_end = b_start + baseline_duration

            s_samp = max(0, int(b_start * fs))
            e_samp = min(data.shape[1], int(b_end * fs))
            if e_samp > s_samp:
                baseline = data[:, s_samp:e_samp]
                # 太短则重复填充
                target = int(baseline_duration * fs)
                if baseline.shape[1] < target:
                    n_rep = int(np.ceil(target / baseline.shape[1]))
                    baseline = np.tile(baseline, (1, n_rep))[:, :target]
                return baseline
        except (ValueError, TypeError, IndexError):
            pass

    # Fallback: 发作前数据
    try:
        sz_start = float(row['sz_start'])
        if sz_start > 10:
            b_end_t = sz_start - 1.0
            b_start_t = max(0, b_end_t - baseline_duration)
            s_samp = int(b_start_t * fs)
            e_samp = int(b_end_t * fs)
            if e_samp > s_samp:
                return data[:, s_samp:e_samp]
    except (KeyError, ValueError, TypeError):
        pass

    return None


# ==============================================================================
# 文件查找 (私有数据)
# ==============================================================================

def find_private_file(
    loc: str, data_roots: List[str], fmt: str = 'set',
) -> Optional[str]:
    """
    根据manifest中的loc字段查找私有数据文件

    SET格式: 根目录+'_processed', 文件名+'_filtered_3_45_postICA_eye.set'
    EDF格式: 直接在根目录下查找
    """
    base = Path(loc)
    stem = base.stem
    parent = base.parent

    for root in data_roots:
        if fmt == 'set':
            root_p = str(root) + '_processed'
            new_fn = f"{stem}_filtered_3_45_postICA_eye.set"

            # 标准路径
            for candidate in [
                Path(root_p) / parent / new_fn,
                Path(root_p) / new_fn,
            ]:
                if candidate.exists():
                    return str(candidate)

            # 递归搜索
            for found in Path(root_p).rglob(new_fn):
                return str(found)

            # Fallback: 不带后缀
            simple = Path(root_p) / parent / (stem + '.set')
            if simple.exists():
                return str(simple)
        else:
            rp = Path(root)
            full = rp / loc
            if full.exists():
                return str(full)
            for ext in ['.edf', '.EDF']:
                test = rp / parent / (stem + ext)
                if test.exists():
                    return str(test)
            for found in rp.rglob(stem + '.edf'):
                return str(found)

    return None


# ==============================================================================
# 单文件处理: TUSZ
# ==============================================================================

def process_tusz_file(row: pd.Series, cfg: PreprocessConfig) -> List[Dict]:
    """
    处理TUSZ manifest中的一行 (一个EDF文件)

    流程: 读取 → 提取电极 → 滤波/裁剪/重采样 → 提取发作段 → TCP转换 → 标准化 → 窗口

    Returns:
        样本字典列表 (每个窗口一个)
    """
    results = []

    # --- 定位EDF文件 ---
    edf_rel = row['edf_path']
    edf_path = str(Path(cfg.tusz_data_root) / edf_rel)
    if not Path(edf_path).exists():
        logger.warning(f"文件不存在: {edf_path}")
        return results

    # --- 1. 读取 ---
    try:
        raw_data, ch_names, fs = read_data_file(edf_path)
    except Exception as e:
        logger.error(f"读取失败 {edf_path}: {e}")
        return results

    # --- 2. 提取标准电极 ---
    electrode_data, name_to_idx = extract_standard_electrodes(raw_data, ch_names)

    # --- 3-5. 滤波/裁剪/重采样 ---
    electrode_data, cur_fs = preprocess_electrode_data(electrode_data, fs, cfg)

    # --- 标签 ---
    ch_labels = get_tusz_channel_labels(row)
    reg_labels = get_tusz_region_labels(row)

    # --- 解析发作时间 ---
    sz_starts_str = row.get('sz_starts', '')
    sz_ends_str = row.get('sz_ends', '')
    if pd.isna(sz_starts_str) or not str(sz_starts_str).strip():
        return results

    try:
        sz_starts = [float(t.strip()) for t in str(sz_starts_str).split(';') if t.strip()]
        sz_ends = [float(t.strip()) for t in str(sz_ends_str).split(';') if t.strip()]
    except (ValueError, TypeError):
        return results

    if not sz_starts or not sz_ends or len(sz_starts) != len(sz_ends):
        return results

    file_dur = electrode_data.shape[1] / cur_fs

    # --- 处理每个发作事件 ---
    for sz_idx, (sz_s, sz_e) in enumerate(zip(sz_starts, sz_ends)):
        if sz_e - sz_s < cfg.min_seizure_duration:
            continue

        # 6. 提取发作段（带缓冲）
        seg_s = max(0, sz_s - cfg.pre_seizure_buffer)
        seg_e = min(file_dur, sz_e + cfg.post_seizure_buffer)
        seizure_seg = electrode_data[:, int(seg_s * cur_fs):int(seg_e * cur_fs)]

        # 提取基线
        baseline_seg = extract_tusz_baseline(
            electrode_data, cur_fs, sz_starts, sz_ends, sz_idx, cfg.baseline_duration
        )

        # 7. TCP转换
        sz_tcp, ch_mask = convert_to_tcp_bipolar(seizure_seg, name_to_idx)

        # 8. 标准化
        has_bl = False
        bl_mu, bl_sd = None, None
        bl_tcp = None

        if baseline_seg is not None and baseline_seg.shape[1] > 0:
            bl_tcp, _ = convert_to_tcp_bipolar(baseline_seg, name_to_idx)
            bl_mu, bl_sd = compute_baseline_stats(bl_tcp)
            sz_tcp = normalize_by_baseline(sz_tcp, bl_mu, bl_sd)
            has_bl = True
        else:
            sz_tcp = normalize_zscore(sz_tcp)

        # 9. 窗口分割
        windows = segment_into_windows(sz_tcp, cfg.target_fs, cfg.window_len, cfg.window_overlap)

        patient_id = str(row['patient_id'])
        file_id = str(row['file_id'])
        split = str(row.get('split', 'train'))

        for w_idx in range(windows.shape[0]):
            results.append({
                'eeg_data': windows[w_idx].astype(np.float32),
                'channel_labels': ch_labels,
                'region_labels': reg_labels,
                'channel_mask': ch_mask,
                'is_seizure': True,
                'patient_id': patient_id,
                'file_id': file_id,
                'seizure_idx': sz_idx,
                'window_idx': w_idx,
                'data_type': 'tusz',
                'split': split,
            })

        # 基线窗口（全0标签）
        if cfg.include_baseline and has_bl and bl_tcp is not None:
            bl_norm = normalize_by_baseline(bl_tcp, bl_mu, bl_sd)
            bl_windows = segment_into_windows(bl_norm, cfg.target_fs, cfg.window_len, cfg.window_overlap)

            zero_ch = np.zeros(N_TCP_CHANNELS, dtype=np.float32)
            zero_reg = np.zeros(N_REGIONS, dtype=np.float32)

            for w_idx in range(bl_windows.shape[0]):
                results.append({
                    'eeg_data': bl_windows[w_idx].astype(np.float32),
                    'channel_labels': zero_ch,
                    'region_labels': zero_reg,
                    'channel_mask': ch_mask,
                    'is_seizure': False,
                    'patient_id': patient_id,
                    'file_id': file_id,
                    'seizure_idx': sz_idx,
                    'window_idx': w_idx,
                    'data_type': 'tusz',
                    'split': split,
                })

    return results


# ==============================================================================
# 单文件处理: 私有数据
# ==============================================================================

def process_private_file(row: pd.Series, cfg: PreprocessConfig) -> List[Dict]:
    """
    处理私有数据manifest中的一行

    与TUSZ的差异:
    - 文件查找逻辑不同 (.set格式特殊路径)
    - 有坏段剔除 (mask_segments)
    - 基线来源不同 (base_line列 / 发作前数据)
    - 标签从per-electrode SOZ列推导
    """
    results = []

    # --- 定位文件 ---
    loc = row.get('loc', '')
    filepath = find_private_file(loc, cfg.private_data_roots, cfg.private_file_format)
    if filepath is None:
        logger.warning(f"找不到文件: {loc}")
        return results

    # --- 1. 读取 ---
    try:
        raw_data, ch_names, fs = read_data_file(filepath)
    except Exception as e:
        logger.error(f"读取失败 {filepath}: {e}")
        return results

    # --- 2. 提取标准电极 ---
    electrode_data, name_to_idx = extract_standard_electrodes(raw_data, ch_names)

    # --- 标签 ---
    ch_labels = get_private_channel_labels(row)
    reg_labels = get_private_region_labels(row)

    # --- 解析发作时间 ---
    try:
        sz_start = float(row['sz_start'])
        sz_end = float(row['sz_end'])
    except (KeyError, ValueError, TypeError):
        logger.warning(f"无效发作时间: {row.get('fn', '?')}")
        return results

    if sz_end <= sz_start:
        return results

    # --- 3-5. 滤波/裁剪/重采样 ---
    electrode_data, cur_fs = preprocess_electrode_data(electrode_data, fs, cfg)

    # --- 6. 提取发作段 ---
    s_samp = max(0, int(sz_start * cur_fs))
    e_samp = min(electrode_data.shape[1], int(sz_end * cur_fs))
    seizure_seg = electrode_data[:, s_samp:e_samp]

    # 坏段剔除
    mask_segs = parse_mask_segments(row.get('mask_segments', None))
    if mask_segs:
        seizure_seg, clean_dur = remove_bad_segments(
            seizure_seg, cur_fs, mask_segs,
            seg_start=sz_start, seg_end=sz_end
        )
        if seizure_seg.shape[1] == 0:
            logger.warning(f"坏段剔除后无数据: {row.get('fn', '?')}")
            return results

    # 提取基线
    baseline_seg = extract_private_baseline(electrode_data, cur_fs, row, cfg.baseline_duration)

    # --- 7. TCP转换 ---
    sz_tcp, ch_mask = convert_to_tcp_bipolar(seizure_seg, name_to_idx)

    # --- 8. 标准化 ---
    has_bl = False
    bl_mu, bl_sd = None, None
    bl_tcp = None

    if baseline_seg is not None and baseline_seg.shape[1] > 0:
        bl_tcp, _ = convert_to_tcp_bipolar(baseline_seg, name_to_idx)
        bl_mu, bl_sd = compute_baseline_stats(bl_tcp)
        sz_tcp = normalize_by_baseline(sz_tcp, bl_mu, bl_sd)
        has_bl = True
    else:
        sz_tcp = normalize_zscore(sz_tcp)

    # --- 9. 窗口分割 ---
    windows = segment_into_windows(sz_tcp, cfg.target_fs, cfg.window_len, cfg.window_overlap)

    pt_id = str(row.get('pt_id', row.get('patient_id', 'unknown')))
    fn = str(row.get('fn', 'unknown'))
    sz_idx = int(row.get('sz_idx', 0))

    for w_idx in range(windows.shape[0]):
        results.append({
            'eeg_data': windows[w_idx].astype(np.float32),
            'channel_labels': ch_labels,
            'region_labels': reg_labels,
            'channel_mask': ch_mask,
            'is_seizure': True,
            'patient_id': pt_id,
            'file_id': fn,
            'seizure_idx': sz_idx,
            'window_idx': w_idx,
            'data_type': 'private',
            'split': 'train',
        })

    # 基线窗口
    if cfg.include_baseline and has_bl and bl_tcp is not None:
        bl_norm = normalize_by_baseline(bl_tcp, bl_mu, bl_sd)
        bl_windows = segment_into_windows(bl_norm, cfg.target_fs, cfg.window_len, cfg.window_overlap)

        zero_ch = np.zeros(N_TCP_CHANNELS, dtype=np.float32)
        zero_reg = np.zeros(N_REGIONS, dtype=np.float32)

        for w_idx in range(bl_windows.shape[0]):
            results.append({
                'eeg_data': bl_windows[w_idx].astype(np.float32),
                'channel_labels': zero_ch,
                'region_labels': zero_reg,
                'channel_mask': ch_mask,
                'is_seizure': False,
                'patient_id': pt_id,
                'file_id': fn,
                'seizure_idx': sz_idx,
                'window_idx': w_idx,
                'data_type': 'private',
                'split': 'train',
            })

    return results


# ==============================================================================
# 统一处理: combined_manifest.csv
# ==============================================================================

# combined_manifest.csv 的22个TCP通道列名 (与config.py TCP_BIPOLAR_PAIRS对齐)
COMBINED_TCP_COLUMNS = [
    'FP1_F7', 'F7_T3', 'T3_T5', 'T5_O1',       # 左颞链 0-3
    'FP2_F8', 'F8_T4', 'T4_T6', 'T6_O2',       # 右颞链 4-7
    'FP1_F3', 'F3_C3', 'C3_P3', 'P3_O1',       # 左副矢状链 8-11
    'FP2_F4', 'F4_C4', 'C4_P4', 'P4_O2',       # 右副矢状链 12-15
    'A1_T3', 'T3_C3', 'C3_CZ', 'CZ_C4',        # 中央链 16-21
    'C4_T4', 'T4_A2',
]

# combined_manifest列顺序 → config.py TCP_BIPOLAR_PAIRS 顺序 的映射
# combined_manifest: 左颞(0-3), 右颞(4-7), 左副矢(8-11), 右副矢(12-15), 中央(16-21)
# config.py:        左颞(0-3), 右颞(4-7), 中央(8-13), 左副矢(14-17), 右副矢(18-21)
_CSV_TO_CFG_ORDER = [
    0, 1, 2, 3,       # 左颞 → 左颞
    4, 5, 6, 7,       # 右颞 → 右颞
    16, 17, 18, 19, 20, 21,  # 中央 (CSV idx 16-21 → cfg idx 8-13)
    8, 9, 10, 11,     # 左副矢 (CSV idx 8-11 → cfg idx 14-17)
    12, 13, 14, 15,   # 右副矢 (CSV idx 12-15 → cfg idx 18-21)
]


def get_combined_channel_labels(row: pd.Series) -> np.ndarray:
    """
    从 combined_manifest.csv 行直接读取22通道SOZ标签

    combined_manifest 的列顺序与 config.py 的 TCP_BIPOLAR_PAIRS 不同，
    需要重新映射到 config.py 的顺序。

    Returns:
        (22,) float32 — 按 config.py TCP_BIPOLAR_PAIRS 顺序排列
    """
    # 先按 CSV 列顺序读取
    csv_labels = np.array(
        [int(float(row.get(col, 0))) for col in COMBINED_TCP_COLUMNS],
        dtype=np.float32,
    )  # (22,) 按CSV列顺序

    # 映射到 config.py TCP 顺序
    cfg_labels = np.zeros(N_TCP_CHANNELS, dtype=np.float32)
    for cfg_idx, csv_idx in enumerate(_CSV_TO_CFG_ORDER):
        cfg_labels[cfg_idx] = csv_labels[csv_idx]

    return cfg_labels


def process_combined_row(row: pd.Series, cfg: PreprocessConfig) -> List[Dict]:
    """
    处理 combined_manifest.csv 中的一行

    根据 source 列自动分发到 TUSZ 或私有数据的处理逻辑。
    标签统一从22个TCP列直接读取。

    Args:
        row: combined_manifest.csv 中的一行
        cfg: 预处理配置

    Returns:
        样本字典列表 (每个窗口一个)
    """
    source = str(row.get('source', 'tusz')).strip().lower()
    results = []

    # ── 标签（统一从22列读取）──
    ch_labels = get_combined_channel_labels(row)
    reg_labels = get_channel_to_region_labels(ch_labels)

    # ── 解析发作时间 ──
    try:
        sz_start = float(row['sz_start'])
        sz_end = float(row['sz_end'])
    except (KeyError, ValueError, TypeError):
        logger.debug(f"无效发作时间: patient={row.get('patient_id', '?')}")
        return results
    if sz_end <= sz_start:
        return results

    # ── 定位文件 ──
    edf_rel = str(row.get('edf_path', ''))
    if source == 'tusz':
        edf_path = str(Path(cfg.tusz_data_root) / edf_rel)
    else:
        # 私有数据: 尝试多个数据根目录
        edf_path = None
        for root in cfg.private_data_roots:
            candidate = Path(root) / edf_rel
            if candidate.exists():
                edf_path = str(candidate)
                break
        if edf_path is None:
            # 尝试 find_private_file (兼容 .set 路径)
            edf_path = find_private_file(edf_rel, cfg.private_data_roots, cfg.private_file_format)
        if edf_path is None:
            logger.warning(f"找不到文件: {edf_rel} (source={source})")
            return results

    if not Path(edf_path).exists():
        logger.warning(f"文件不存在: {edf_path}")
        return results

    # ── 1. 读取 ──
    try:
        raw_data, ch_names, fs = read_data_file(edf_path)
    except Exception as e:
        logger.error(f"读取失败 {edf_path}: {e}")
        return results

    # ── 2. 提取标准电极 ──
    electrode_data, name_to_idx = extract_standard_electrodes(raw_data, ch_names)

    # ── 3-5. 滤波/裁剪/重采样 ──
    electrode_data, cur_fs = preprocess_electrode_data(electrode_data, fs, cfg)

    file_dur = electrode_data.shape[1] / cur_fs

    # ── 6. 提取发作段 ──
    if source == 'tusz':
        # TUSZ: 带缓冲区
        seg_s = max(0, sz_start - cfg.pre_seizure_buffer)
        seg_e = min(file_dur, sz_end + cfg.post_seizure_buffer)
    else:
        # 私有数据: 精确发作段
        seg_s = max(0, sz_start)
        seg_e = min(file_dur, sz_end)

    seizure_seg = electrode_data[:, int(seg_s * cur_fs):int(seg_e * cur_fs)]

    # 私有数据坏段剔除
    if source == 'private':
        mask_segs = parse_mask_segments(row.get('mask_segments', None))
        if mask_segs:
            seizure_seg, _ = remove_bad_segments(
                seizure_seg, cur_fs, mask_segs,
                seg_start=seg_s, seg_end=seg_e,
            )
            if seizure_seg.shape[1] == 0:
                logger.debug(f"坏段剔除后无数据: {edf_path}")
                return results

    # ── 提取基线 ──
    baseline_seg = None
    if source == 'tusz':
        # TUSZ: 从非发作间隔提取（简化：只用一个发作事件）
        baseline_seg = extract_tusz_baseline(
            electrode_data, cur_fs,
            [sz_start], [sz_end], 0,
            cfg.baseline_duration,
        )
    else:
        baseline_seg = extract_private_baseline(
            electrode_data, cur_fs, row, cfg.baseline_duration,
        )

    # ── 7. TCP转换 ──
    sz_tcp, ch_mask = convert_to_tcp_bipolar(seizure_seg, name_to_idx)

    # ── 8. 标准化 ──
    has_bl = False
    bl_mu, bl_sd = None, None
    bl_tcp = None

    if baseline_seg is not None and baseline_seg.shape[1] > 0:
        bl_tcp, _ = convert_to_tcp_bipolar(baseline_seg, name_to_idx)
        bl_mu, bl_sd = compute_baseline_stats(bl_tcp)
        sz_tcp = normalize_by_baseline(sz_tcp, bl_mu, bl_sd)
        has_bl = True
    else:
        sz_tcp = normalize_zscore(sz_tcp)

    # ── 9. 窗口分割 ──
    windows = segment_into_windows(sz_tcp, cfg.target_fs, cfg.window_len, cfg.window_overlap)

    patient_id = str(row.get('patient_id', 'unknown'))
    split = str(row.get('split', 'train'))
    # 用 edf_path 的 stem 作为 file_id
    file_id = Path(edf_rel).stem if edf_rel else 'unknown'

    for w_idx in range(windows.shape[0]):
        results.append({
            'eeg_data': windows[w_idx].astype(np.float32),
            'channel_labels': ch_labels,
            'region_labels': reg_labels,
            'channel_mask': ch_mask,
            'is_seizure': True,
            'patient_id': patient_id,
            'file_id': file_id,
            'seizure_idx': 0,
            'window_idx': w_idx,
            'data_type': source,
            'split': split,
        })

    # ── 基线窗口 ──
    if cfg.include_baseline and has_bl and bl_tcp is not None:
        bl_norm = normalize_by_baseline(bl_tcp, bl_mu, bl_sd)
        bl_windows = segment_into_windows(
            bl_norm, cfg.target_fs, cfg.window_len, cfg.window_overlap,
        )

        zero_ch = np.zeros(N_TCP_CHANNELS, dtype=np.float32)
        zero_reg = np.zeros(N_REGIONS, dtype=np.float32)

        for w_idx in range(bl_windows.shape[0]):
            results.append({
                'eeg_data': bl_windows[w_idx].astype(np.float32),
                'channel_labels': zero_ch,
                'region_labels': zero_reg,
                'channel_mask': ch_mask,
                'is_seizure': False,
                'patient_id': patient_id,
                'file_id': file_id,
                'seizure_idx': 0,
                'window_idx': w_idx,
                'data_type': source,
                'split': split,
            })

    return results


def preprocess_combined(cfg: PreprocessConfig) -> List[Dict]:
    """
    使用 combined_manifest.csv 统一处理 TUSZ + 私有数据

    根据每行的 source 列自动选择处理逻辑和数据根目录。

    保存结构:
        {output_root}/{source}/{patient_id}/{patient_id}_{file}_sz0_w0.npz

    Args:
        cfg: PreprocessConfig (必须设置 combined_manifest)

    Returns:
        index_rows: 所有保存样本的索引信息列表
    """
    logger.info("=" * 60)
    logger.info("统一处理 combined_manifest.csv")
    logger.info("=" * 60)
    logger.info(f"  Manifest:    {cfg.combined_manifest}")
    logger.info(f"  TUSZ根:      {cfg.tusz_data_root}")
    logger.info(f"  私有数据根:   {cfg.private_data_roots}")
    logger.info(f"  输出:        {cfg.output_root}")
    logger.info(f"  参数: fs={cfg.target_fs}Hz, filter={cfg.highpass_fc}-{cfg.lowpass_fc}Hz, "
                f"clip=+/-{cfg.clip_n_std}std, window={cfg.window_len}s@{cfg.window_overlap*100:.0f}%")

    df = pd.read_csv(cfg.combined_manifest)
    logger.info(f"  总记录: {len(df)}")

    # 按source统计
    for src in df['source'].unique():
        sub = df[df['source'] == src]
        logger.info(f"    {src}: {len(sub)} 行, {sub['patient_id'].nunique()} 患者")

    out_root = Path(cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict] = []
    stats = {'total': 0, 'ok': 0, 'fail': 0, 'tusz_ok': 0, 'private_ok': 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Combined"):
        source = str(row.get('source', 'tusz')).strip().lower()

        try:
            samples = process_combined_row(row, cfg)
            if samples:
                # 保存到 {output_root}/{source}/ 目录
                out_dir = out_root / source
                out_dir.mkdir(parents=True, exist_ok=True)

                for s in samples:
                    idx_row = _save_sample_to_patient_dir(s, out_dir, cfg.output_root)
                    if idx_row:
                        index_rows.append(idx_row)
                    stats['total'] += 1

                stats['ok'] += 1
                if source == 'tusz':
                    stats['tusz_ok'] += 1
                else:
                    stats['private_ok'] += 1
            else:
                stats['fail'] += 1
        except Exception as e:
            logger.error(f"异常 patient={row.get('patient_id', '?')}: {e}")
            stats['fail'] += 1

    logger.info(f"\n统一处理完成:")
    logger.info(f"  文件成功={stats['ok']} (TUSZ={stats['tusz_ok']}, "
                f"Private={stats['private_ok']}), 失败={stats['fail']}, 样本={stats['total']}")

    return index_rows


# ==============================================================================
# 主处理流程 (旧模式 — 分别处理)
# ==============================================================================

def preprocess_tusz(cfg: PreprocessConfig) -> List[Dict]:
    """
    处理TUSZ公共数据集

    保存结构:
        F:/process_dataset/tusz/{patient_id}/{patient_id}_{file_id}_sz{i}_w{j}.npz

    Returns:
        index_rows: 所有保存样本的索引信息列表
    """
    logger.info("=" * 60)
    logger.info("处理TUSZ数据")
    logger.info("=" * 60)
    logger.info(f"  Manifest: {cfg.tusz_manifest}")
    logger.info(f"  数据根: {cfg.tusz_data_root}")
    logger.info(f"  输出: {cfg.output_dir}")
    logger.info(f"  参数: fs={cfg.target_fs}Hz, filter={cfg.highpass_fc}-{cfg.lowpass_fc}Hz, "
                f"clip=±{cfg.clip_n_std}std, window={cfg.window_len}s@{cfg.window_overlap*100:.0f}%")

    df = pd.read_csv(cfg.tusz_manifest)
    logger.info(f"  总记录: {len(df)}")

    df_sz = df[df['has_seizure'] == True].reset_index(drop=True)
    logger.info(f"  含发作: {len(df_sz)}")

    if len(df_sz) == 0:
        logger.warning("没有含发作的记录!")
        return []

    out_path = Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict] = []
    total, ok, fail = 0, 0, 0

    for _, row in tqdm(df_sz.iterrows(), total=len(df_sz), desc="TUSZ"):
        try:
            samples = process_tusz_file(row, cfg)
            if samples:
                for s in samples:
                    idx_row = _save_sample_to_patient_dir(s, out_path, cfg.output_root)
                    if idx_row:
                        index_rows.append(idx_row)
                    total += 1
                ok += 1
            else:
                fail += 1
        except Exception as e:
            logger.error(f"异常 {row.get('file_id', '?')}: {e}")
            fail += 1

    logger.info(f"TUSZ完成: 文件成功={ok}, 失败={fail}, 样本={total}")
    return index_rows


def preprocess_private(cfg: PreprocessConfig) -> List[Dict]:
    """
    处理私有数据集

    保存结构:
        F:/process_dataset/private/{patient_id}/{patient_id}_{file_id}_sz{i}_w{j}.npz

    Returns:
        index_rows: 所有保存样本的索引信息列表
    """
    logger.info("=" * 60)
    logger.info("处理私有数据")
    logger.info("=" * 60)
    logger.info(f"  Manifest: {cfg.private_manifest}")
    logger.info(f"  数据根: {cfg.private_data_roots}")
    logger.info(f"  输出: {cfg.output_dir}")
    logger.info(f"  格式: {cfg.private_file_format}")
    logger.info(f"  参数: fs={cfg.target_fs}Hz, filter={cfg.highpass_fc}-{cfg.lowpass_fc}Hz, "
                f"clip=±{cfg.clip_n_std}std, window={cfg.window_len}s@{cfg.window_overlap*100:.0f}%")

    df = pd.read_csv(cfg.private_manifest)
    logger.info(f"  总记录: {len(df)}")

    out_path = Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict] = []
    total, ok, fail = 0, 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="私有数据"):
        try:
            samples = process_private_file(row, cfg)
            if samples:
                for s in samples:
                    idx_row = _save_sample_to_patient_dir(s, out_path, cfg.output_root)
                    if idx_row:
                        index_rows.append(idx_row)
                    total += 1
                ok += 1
            else:
                fail += 1
        except Exception as e:
            logger.error(f"异常 {row.get('fn', '?')}: {e}")
            fail += 1

    logger.info(f"私有数据完成: 文件成功={ok}, 失败={fail}, 样本={total}")
    return index_rows


# ==============================================================================
# 保存辅助
# ==============================================================================

def _save_sample_to_patient_dir(
    sample: Dict, data_dir: Path, output_root: str,
) -> Optional[Dict]:
    """
    保存单个样本到按patient_id分组的子目录

    保存路径: {data_dir}/{patient_id}/{filename}.npz

    Args:
        sample:     处理好的样本字典
        data_dir:   数据类型目录 (如 F:/process_dataset/tusz)
        output_root: 总根目录 (如 F:/process_dataset)

    Returns:
        索引行字典 (用于生成index CSV)
    """
    pt = str(sample['patient_id']).replace('/', '_').replace('\\', '_').replace(' ', '_')
    fid = str(sample['file_id']).replace('/', '_').replace('\\', '_').replace(' ', '_')
    si = sample['seizure_idx']
    wi = sample['window_idx']
    tag = 'sz' if sample['is_seizure'] else 'bl'
    dtype = sample['data_type']
    split = sample.get('split', 'train')

    # 按patient_id分组
    patient_dir = data_dir / pt
    patient_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{pt}_{fid}_{tag}{si}_w{wi}.npz"
    npz_path = patient_dir / fname

    np.savez_compressed(
        str(npz_path),
        eeg_data=sample['eeg_data'],               # (22, window_samples)
        channel_labels=sample['channel_labels'],    # (22,)
        region_labels=sample['region_labels'],       # (5,)
        channel_mask=sample['channel_mask'],         # (22,)
        is_seizure=np.bool_(sample['is_seizure']),
        patient_id=str(pt),
        file_id=str(fid),
        seizure_idx=np.int32(si),
        window_idx=np.int32(wi),
        data_type=str(dtype),
        split=str(split),
    )

    # 构造索引行
    ch_labels = sample['channel_labels']
    reg_labels = sample['region_labels']
    root_path = Path(output_root)

    try:
        rel_path = str(npz_path.relative_to(root_path))
    except ValueError:
        rel_path = str(npz_path)

    soz_channel_names = ','.join(
        TCP_CHANNEL_NAMES[i] for i in range(N_TCP_CHANNELS)
        if ch_labels[i] > 0
    )
    soz_region_names = ','.join(
        BRAIN_REGIONS[i] for i in range(N_REGIONS)
        if reg_labels[i] > 0
    )

    return {
        'npz_path': str(npz_path),
        'npz_rel': rel_path,
        'source': dtype,
        'patient_id': pt,
        'file_id': fid,
        'seizure_idx': si,
        'window_idx': wi,
        'is_seizure': int(sample['is_seizure']),
        'has_soz': int(np.any(ch_labels > 0)),
        'n_soz_channels': int(np.sum(ch_labels > 0)),
        'soz_channels': soz_channel_names,
        'n_soz_regions': int(np.sum(reg_labels > 0)),
        'soz_regions': soz_region_names,
        'split': split,
    }


def _save_global_metadata(cfg: PreprocessConfig):
    """
    保存全局元数据到 F:/process_dataset/meta/ 目录

    包含:
        - adj_matrix.npy:     22×22 空间邻接矩阵
        - tcp_channels.json:  TCP通道名列表
        - std_electrodes.json: 标准21电极列表
        - brain_regions.json: 脑区列表
        - tcp_chains.json:    链定义
        - config.json:        预处理配置
    """
    meta_dir = Path(cfg.meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 邻接矩阵
    adj_path = meta_dir / 'adj_matrix.npy'
    adj = build_tcp_adjacency_matrix()
    np.save(str(adj_path), adj)
    logger.info(f"  邻接矩阵: {adj_path} shape={adj.shape}")

    # TCP通道名
    tcp_path = meta_dir / 'tcp_channels.json'
    with open(str(tcp_path), 'w', encoding='utf-8') as f:
        json.dump(TCP_CHANNEL_NAMES, f, indent=2)

    # 标准21电极
    elec_path = meta_dir / 'std_electrodes.json'
    with open(str(elec_path), 'w', encoding='utf-8') as f:
        json.dump(STANDARD_21_ELECTRODES, f, indent=2)

    # 脑区
    region_path = meta_dir / 'brain_regions.json'
    with open(str(region_path), 'w', encoding='utf-8') as f:
        json.dump(BRAIN_REGIONS, f, indent=2)

    # 链定义
    chain_path = meta_dir / 'tcp_chains.json'
    with open(str(chain_path), 'w', encoding='utf-8') as f:
        json.dump(TCP_CHAINS, f, indent=2)

    # TCP通道→脑区映射
    region_map_path = meta_dir / 'tcp_to_region.json'
    with open(str(region_map_path), 'w', encoding='utf-8') as f:
        json.dump(TCP_TO_REGION, f, indent=2, ensure_ascii=False)

    # TCP通道→半球映射
    hemi_map_path = meta_dir / 'tcp_to_hemisphere.json'
    with open(str(hemi_map_path), 'w', encoding='utf-8') as f:
        json.dump(TCP_TO_HEMISPHERE, f, indent=2)

    # 预处理配置
    config_dict = {
        'target_fs': cfg.target_fs,
        'highpass_fc': cfg.highpass_fc,
        'lowpass_fc': cfg.lowpass_fc,
        'filter_order': cfg.filter_order,
        'clip_n_std': cfg.clip_n_std,
        'window_len': cfg.window_len,
        'window_overlap': cfg.window_overlap,
        'window_samples': int(cfg.window_len * cfg.target_fs),
        'normalize_method': cfg.normalize_method,
        'pre_seizure_buffer': cfg.pre_seizure_buffer,
        'post_seizure_buffer': cfg.post_seizure_buffer,
        'baseline_duration': cfg.baseline_duration,
        'n_tcp_channels': N_TCP_CHANNELS,
        'n_regions': N_REGIONS,
        'tcp_channel_names': TCP_CHANNEL_NAMES,
        'brain_regions': BRAIN_REGIONS,
    }
    cfg_path = meta_dir / 'config.json'
    with open(str(cfg_path), 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"  全局元数据已保存到: {meta_dir}")


def _save_index_csv(
    index_rows: List[Dict],
    output_root: str,
    label: str = 'all',
):
    """
    保存索引CSV到 F:/process_dataset/index_{label}.csv

    Args:
        index_rows: 索引行列表
        output_root: 总根目录
        label: 'tusz' / 'private' / 'all'
    """
    if not index_rows:
        logger.warning(f"  无{label}样本, 跳过索引CSV")
        return

    root = Path(output_root)
    df = pd.DataFrame(index_rows)
    idx_path = root / f'index_{label}.csv'
    df.to_csv(str(idx_path), index=False)

    n_sz = int(df['is_seizure'].sum()) if 'is_seizure' in df.columns else 0
    n_soz = int(df['has_soz'].sum()) if 'has_soz' in df.columns else 0
    n_patients = df['patient_id'].nunique() if 'patient_id' in df.columns else 0

    logger.info(f"  索引CSV: {idx_path}")
    logger.info(f"    样本={len(df)}, 发作={n_sz}, SOZ+={n_soz}, 患者={n_patients}")


# ==============================================================================
# 训练时加载 — ProcessedEEGDataset
# ==============================================================================

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ProcessedEEGDataset:
    """
    从 F:/process_dataset/ 加载预处理好的 .npz 数据

    目录结构 (由 preprocess.py 生成):
        {root}/
            index_all.csv / index_tusz.csv / index_private.csv
            meta/
                adj_matrix.npy, tcp_channels.json, std_electrodes.json,
                brain_regions.json, tcp_chains.json, config.json
            tusz/{patient_id}/{patient_id}_{file}_sz0_w0.npz
            private/{patient_id}/{patient_id}_{file}_sz0_w0.npz

    每个 .npz 包含:
        eeg_data:       (22, window_samples)  TCP双极数据 float32
        channel_labels: (22,)                 per-channel SOZ标签
        region_labels:  (5,)                  per-region SOZ标签
        channel_mask:   (22,)                 通道有效标记
        is_seizure:     bool                  是否发作窗口

    Usage:
        # 基本使用
        ds = ProcessedEEGDataset('F:/process_dataset')
        eeg, ch_labels, reg_labels, mask, meta = ds[0]

        # 仅TUSZ
        ds = ProcessedEEGDataset('F:/process_dataset', subset='tusz')

        # 按患者划分 train/val/test
        train_pids, val_pids, test_pids = ds.split_by_patient()
        train_ds = ProcessedEEGDataset('F:/process_dataset', patient_ids=train_pids)
        val_ds   = ProcessedEEGDataset('F:/process_dataset', patient_ids=val_pids)

        # PyTorch DataLoader
        loader = train_ds.create_dataloader(batch_size=32, shuffle=True)
        for eeg_batch, ch_labels_batch, reg_labels_batch, mask_batch, metas in loader:
            # eeg_batch: (B, 22, window_samples)
            # ch_labels_batch: (B, 22)
            # reg_labels_batch: (B, 5)
            # mask_batch: (B, 22)
            ...
    """

    def __init__(
        self,
        root: str = r'F:\process_dataset',
        subset: str = 'all',
        patient_ids: Optional[List[str]] = None,
        source_filter: Optional[str] = None,
        seizure_only: bool = False,
        soz_only: bool = False,
        split_filter: Optional[str] = None,
        preload: bool = False,
    ):
        """
        Args:
            root:          预处理数据根目录
            subset:        'all' / 'tusz' / 'private'
            patient_ids:   仅包含这些患者 (用于train/val/test拆分)
            source_filter: 'tusz' / 'private' (按来源过滤)
            seizure_only:  仅保留发作窗口
            soz_only:      仅保留SOZ阳性样本
            split_filter:  'train' / 'dev' / 'eval' (按split过滤)
            preload:       True=全部预加载到内存
        """
        self.root = Path(root)
        self.preload = preload

        # 加载索引CSV
        index_name = f'index_{subset}.csv'
        index_path = self.root / index_name
        if not index_path.exists():
            raise FileNotFoundError(
                f"索引文件不存在: {index_path}\n"
                f"请先运行预处理: python preprocess.py --data-type both"
            )
        self.df = pd.read_csv(str(index_path))

        # 过滤
        if source_filter:
            self.df = self.df[self.df['source'] == source_filter].reset_index(drop=True)
        if patient_ids is not None:
            pid_set = set(str(p) for p in patient_ids)
            self.df = self.df[self.df['patient_id'].astype(str).isin(pid_set)].reset_index(drop=True)
        if seizure_only:
            self.df = self.df[self.df['is_seizure'] == 1].reset_index(drop=True)
        if soz_only:
            self.df = self.df[self.df['has_soz'] == 1].reset_index(drop=True)
        if split_filter:
            self.df = self.df[self.df['split'] == split_filter].reset_index(drop=True)

        # 加载全局元数据
        meta_dir = self.root / 'meta'
        self.adj = None
        self.tcp_channels = TCP_CHANNEL_NAMES  # 默认
        self.brain_regions = BRAIN_REGIONS
        self.config = {}

        if meta_dir.exists():
            adj_path = meta_dir / 'adj_matrix.npy'
            if adj_path.exists():
                self.adj = np.load(str(adj_path))
            tcp_path = meta_dir / 'tcp_channels.json'
            if tcp_path.exists():
                with open(str(tcp_path)) as f:
                    self.tcp_channels = json.load(f)
            region_path = meta_dir / 'brain_regions.json'
            if region_path.exists():
                with open(str(region_path)) as f:
                    self.brain_regions = json.load(f)
            cfg_path = meta_dir / 'config.json'
            if cfg_path.exists():
                with open(str(cfg_path)) as f:
                    self.config = json.load(f)

        if self.adj is None:
            self.adj = build_tcp_adjacency_matrix()

        # 预加载
        self._cache: Optional[List] = None
        if preload:
            self._cache = [self._load_sample(i) for i in range(len(self.df))]

        logger.info(
            f"ProcessedEEGDataset: {len(self.df)} samples from {index_path.name}, "
            f"patients={self.df['patient_id'].nunique()}, "
            f"seizure={int(self.df['is_seizure'].sum())}, "
            f"SOZ+={int(self.df['has_soz'].sum())}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            return self._cache[idx]
        return self._load_sample(idx)

    def _load_sample(self, idx: int):
        """
        加载单个样本

        Returns:
            (eeg_data, channel_labels, region_labels, channel_mask, meta_dict)
        """
        row = self.df.iloc[idx]

        # 解析npz路径
        npz_path = str(row['npz_path'])
        if not Path(npz_path).exists():
            npz_path = str(self.root / row['npz_rel'])

        d = np.load(npz_path, allow_pickle=True)
        eeg_data = d['eeg_data'].astype(np.float32)          # (22, window_samples)
        channel_labels = d['channel_labels'].astype(np.float32)  # (22,)
        region_labels = d['region_labels'].astype(np.float32)    # (5,)
        channel_mask = d['channel_mask'].astype(np.float32)      # (22,)

        meta = {
            'source': str(row['source']),
            'patient_id': str(row['patient_id']),
            'file_id': str(row.get('file_id', '')),
            'is_seizure': int(row['is_seizure']),
            'has_soz': int(row['has_soz']),
            'npz_path': npz_path,
        }

        if _HAS_TORCH:
            return (
                torch.from_numpy(eeg_data),
                torch.from_numpy(channel_labels),
                torch.from_numpy(region_labels),
                torch.from_numpy(channel_mask),
                meta,
            )
        return eeg_data, channel_labels, region_labels, channel_mask, meta

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def get_patient_ids(self) -> List[str]:
        """获取所有患者ID"""
        return sorted(self.df['patient_id'].astype(str).unique().tolist())

    def get_source_counts(self) -> Dict[str, int]:
        """获取各数据源的样本数"""
        return self.df['source'].value_counts().to_dict()

    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        return {
            'n_samples': len(self.df),
            'n_patients': self.df['patient_id'].nunique(),
            'n_seizure': int(self.df['is_seizure'].sum()),
            'n_baseline': int((self.df['is_seizure'] == 0).sum()),
            'n_soz_positive': int(self.df['has_soz'].sum()),
            'source_counts': self.get_source_counts(),
            'n_tcp_channels': len(self.tcp_channels),
            'n_regions': len(self.brain_regions),
            'window_samples': self.config.get('window_samples', 'unknown'),
        }

    def split_by_patient(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        按患者划分 train/val/test，确保同一患者的样本不跨集

        Returns:
            (train_pids, val_pids, test_pids)
        """
        rng = np.random.RandomState(seed)
        pids = sorted(self.df['patient_id'].astype(str).unique().tolist())
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
        """
        创建 PyTorch DataLoader

        Returns:
            DataLoader, 每个batch返回:
                eeg:          (B, 22, window_samples)
                ch_labels:    (B, 22)
                reg_labels:   (B, 5)
                mask:         (B, 22)
                metas:        list[dict]
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch未安装, 请运行: pip install torch")

        def collate_fn(batch):
            eegs, ch_labels, reg_labels, masks, metas = zip(*batch)
            return (
                torch.stack(eegs),           # (B, 22, window_samples)
                torch.stack(ch_labels),      # (B, 22)
                torch.stack(reg_labels),     # (B, 5)
                torch.stack(masks),          # (B, 22)
                list(metas),                 # list of dicts
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


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='EEG预处理 → TCP 22通道双极导联 (TimeFilter SOZ Detection)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输出目录结构 (F:/process_dataset/):
    tusz/                          <- TUSZ公共数据
        {patient_id}/
            {patient_id}_{file}_sz0_w0.npz
            ...
    private/                       <- 私有数据
        {patient_id}/
            {patient_id}_{file}_sz0_w0.npz
            ...
    meta/                          <- 全局元数据
        adj_matrix.npy             <- 22×22 邻接矩阵
        tcp_channels.json          <- TCP通道名
        std_electrodes.json        <- 21标准电极
        brain_regions.json         <- 脑区定义
        tcp_chains.json            <- 链定义
        tcp_to_region.json         <- 通道→脑区映射
        tcp_to_hemisphere.json     <- 通道→半球映射
        config.json                <- 预处理配置
    index_tusz.csv                 <- TUSZ样本索引
    index_private.csv              <- 私有样本索引
    index_all.csv                  <- 全部样本索引

每个 .npz 文件包含:
    eeg_data:       (22, 2400)     TCP双极数据 (float32)
    channel_labels: (22,)          per-channel SOZ标签
    region_labels:  (5,)           per-region SOZ标签
    channel_mask:   (22,)          通道有效标记
    is_seizure:     bool           是否发作窗口
    patient_id, file_id, seizure_idx, window_idx, data_type, split

示例:
    # 处理TUSZ
    python preprocess.py --data-type tusz

    # 处理私有数据
    python preprocess.py --data-type private

    # 同时处理两种数据
    python preprocess.py --data-type both

    # 自定义输出目录
    python preprocess.py --data-type both --output-root D:/my_preprocessed

    # 自定义参数
    python preprocess.py --data-type tusz --target-fs 256 --window-len 10

训练时加载:
    from preprocess import ProcessedEEGDataset
    ds = ProcessedEEGDataset('F:/process_dataset', subset='all')
    loader = ds.create_dataloader(batch_size=32)
        """
    )

    p.add_argument('--data-type', default='private', choices=['tusz', 'private', 'both', 'combined'])
    p.add_argument('--output-root', default=r'F:\process_dataset',
                   help='输出根目录 (默认: F:/process_dataset)')

    # combined_manifest 统一模式
    p.add_argument('--combined-manifest', default=r'E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\TUSZ\combined_manifest.csv',
                   help='combined_manifest.csv 路径 (与 --data-type combined 配合使用)')

    # 预处理参数
    p.add_argument('--target-fs', type=float, default=200.0)
    p.add_argument('--highpass', type=float, default=3.0)
    p.add_argument('--lowpass', type=float, default=45.0)
    p.add_argument('--clip-std', type=float, default=1.0)
    p.add_argument('--window-len', type=float, default=12.0)
    p.add_argument('--window-overlap', type=float, default=0.5)

    # 基线
    p.add_argument('--no-baseline', action='store_true')
    p.add_argument('--baseline-duration', type=float, default=30.0)

    # TUSZ路径
    p.add_argument('--tusz-root', default=r"F:\dataset\TUSZ\v2.0.3\edf")
    p.add_argument('--tusz-manifest', default=None)

    # 私有数据路径
    p.add_argument('--private-roots', nargs='+', default=[r"E:\DataSet\EEG\EEG dataset_SUAT",])
    p.add_argument('--private-manifest', default=None)
    p.add_argument('--private-format', default='edf', choices=['set', 'edf'])

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    # ---- 构建公共配置 ----
    base_cfg_kwargs = dict(
        target_fs=args.target_fs,
        highpass_fc=args.highpass,
        lowpass_fc=args.lowpass,
        clip_n_std=args.clip_std,
        window_len=args.window_len,
        window_overlap=args.window_overlap,
        include_baseline=not args.no_baseline,
        baseline_duration=args.baseline_duration,
        output_root=args.output_root,
    )

    if args.tusz_root:
        base_cfg_kwargs['tusz_data_root'] = args.tusz_root
    if args.tusz_manifest:
        base_cfg_kwargs['tusz_manifest'] = args.tusz_manifest
    if args.private_roots:
        base_cfg_kwargs['private_data_roots'] = args.private_roots
    if args.private_manifest:
        base_cfg_kwargs['private_manifest'] = args.private_manifest
    if args.private_format:
        base_cfg_kwargs['private_file_format'] = args.private_format
    if args.combined_manifest:
        base_cfg_kwargs['combined_manifest'] = args.combined_manifest

    all_index_rows: List[Dict] = []

    # ---- 保存全局元数据 ----
    meta_cfg = PreprocessConfig(**{**base_cfg_kwargs, 'data_type': 'tusz'})
    _save_global_metadata(meta_cfg)

    # ---- combined_manifest 统一模式 ----
    if args.data_type == 'combined':
        if not args.combined_manifest:
            logger.error("使用 --data-type combined 时必须指定 --combined-manifest")
            sys.exit(1)
        combined_cfg = PreprocessConfig(**{**base_cfg_kwargs, 'data_type': 'combined'})
        combined_rows = preprocess_combined(combined_cfg)
        if combined_rows:
            # 按source分开保存索引
            df_idx = pd.DataFrame(combined_rows)
            for src in df_idx['source'].unique():
                src_rows = df_idx[df_idx['source'] == src].to_dict('records')
                _save_index_csv(src_rows, args.output_root, src)
            # 保存合并索引
            _save_index_csv(combined_rows, args.output_root, 'all')
            all_index_rows.extend(combined_rows)

    else:

        # ---- 处理 TUSZ ----
        if args.data_type in ('tusz', 'both'):
            tusz_cfg = PreprocessConfig(**{**base_cfg_kwargs, 'data_type': 'tusz'})
            tusz_rows = preprocess_tusz(tusz_cfg)
            if tusz_rows:
                _save_index_csv(tusz_rows, args.output_root, 'tusz')
                all_index_rows.extend(tusz_rows)

        # ---- 处理 私有数据 ----
        if args.data_type in ('private', 'both'):
            priv_cfg = PreprocessConfig(**{**base_cfg_kwargs, 'data_type': 'private'})
            priv_rows = preprocess_private(priv_cfg)
            if priv_rows:
                _save_index_csv(priv_rows, args.output_root, 'private')
                all_index_rows.extend(priv_rows)

        # ---- 保存合并索引 ----
        if all_index_rows:
            _save_index_csv(all_index_rows, args.output_root, 'all')

    # ---- 打印总结 ----
    logger.info("=" * 60)
    logger.info("预处理全部完成!")
    logger.info(f"  输出根目录: {args.output_root}")
    logger.info(f"  总样本数: {len(all_index_rows)}")
    if all_index_rows:
        df_all = pd.DataFrame(all_index_rows)
        for src in df_all['source'].unique():
            sub = df_all[df_all['source'] == src]
            logger.info(f"    {src}: {len(sub)} 样本, "
                       f"{sub['patient_id'].nunique()} 患者, "
                       f"{int(sub['is_seizure'].sum())} 发作, "
                       f"{int(sub['has_soz'].sum())} SOZ+")
    logger.info("=" * 60)
    logger.info(
        f"\n训练时加载数据:\n"
        f"  from preprocess import ProcessedEEGDataset\n"
        f"  ds = ProcessedEEGDataset('{args.output_root}')\n"
        f"  loader = ds.create_dataloader(batch_size=32)\n"
    )


if __name__ == '__main__':
    main()
