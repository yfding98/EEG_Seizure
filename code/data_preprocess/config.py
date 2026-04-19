#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理配置模块

基于TUSZ v2.0.3官方TCP AR导联定义（01_tcp_ar_montage.txt）:
    共22通道双极导联，分为5条链:
    - 左颞链   (0-3):   FP1-F7, F7-T3, T3-T5, T5-O1
    - 右颞链   (4-7):   FP2-F8, F8-T4, T4-T6, T6-O2
    - 中央链   (8-13):  A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2
    - 左副矢状链(14-17): FP1-F3, F3-C3, C3-P3, P3-O1
    - 右副矢状链(18-21): FP2-F4, F4-C4, C4-P4, P4-O2

TUSZ数据集支持的导联方案:
    - 01_tcp_ar:   22通道 (AR参考, 通道命名 EEG XX-REF)
    - 02_tcp_le:   22通道 (LE参考, 通道命名 EEG XX-LE)
    - 03_tcp_ar_a: 20通道 (AR参考, 无A1-T3和T4-A2)

私有数据(raw)通道情况 (133例, 100%出现率):
    EEG: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6,
         A1, A2, Fz, Cz, Pz, Sph-R, Sph-L
    非EEG: ECG, EMG1, EMG2, PR
    → 所有TCP所需电极(含A1, A2)均可用, 22通道全部可计算
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


# ==============================================================================
# TCP 22通道双极导联定义 (严格按TUSZ官方顺序)
# ==============================================================================

TCP_BIPOLAR_PAIRS = [
    # ---- 左颞链 (Left Temporal Chain) ---- channels 0-3
    ('FP1', 'F7'),   # 0
    ('F7',  'T3'),   # 1
    ('T3',  'T5'),   # 2
    ('T5',  'O1'),   # 3
    # ---- 右颞链 (Right Temporal Chain) ---- channels 4-7
    ('FP2', 'F8'),   # 4
    ('F8',  'T4'),   # 5
    ('T4',  'T6'),   # 6
    ('T6',  'O2'),   # 7
    # ---- 中央链 (Central Chain) ---- channels 8-13
    ('A1',  'T3'),   # 8
    ('T3',  'C3'),   # 9
    ('C3',  'CZ'),   # 10
    ('CZ',  'C4'),   # 11
    ('C4',  'T4'),   # 12
    ('T4',  'A2'),   # 13
    # ---- 左副矢状链 (Left Parasagittal Chain) ---- channels 14-17
    ('FP1', 'F3'),   # 14
    ('F3',  'C3'),   # 15
    ('C3',  'P3'),   # 16
    ('P3',  'O1'),   # 17
    # ---- 右副矢状链 (Right Parasagittal Chain) ---- channels 18-21
    ('FP2', 'F4'),   # 18
    ('F4',  'C4'),   # 19
    ('C4',  'P4'),   # 20
    ('P4',  'O2'),   # 21
]

TCP_CHANNEL_NAMES = [f"{a}-{b}" for a, b in TCP_BIPOLAR_PAIRS]

N_TCP_CHANNELS = len(TCP_BIPOLAR_PAIRS)  # 22

# TCP通道索引（便于代码引用）
TCP_IDX = {name: i for i, name in enumerate(TCP_CHANNEL_NAMES)}


# ==============================================================================
# 链（chain）定义 — 用于TimeFilter的空间分组
# ==============================================================================

TCP_CHAINS = {
    'left_temporal':      [0, 1, 2, 3],               # FP1-F7 → T5-O1
    'right_temporal':     [4, 5, 6, 7],               # FP2-F8 → T6-O2
    'central':            [8, 9, 10, 11, 12, 13],     # A1-T3 → T4-A2
    'left_parasagittal':  [14, 15, 16, 17],           # FP1-F3 → P3-O1
    'right_parasagittal': [18, 19, 20, 21],           # FP2-F4 → P4-O2
}


# ==============================================================================
# 邻接矩阵构建
# ==============================================================================

def build_tcp_adjacency_matrix() -> np.ndarray:
    """
    构建TCP 22通道的空间邻接矩阵（基于共享电极）

    两个双极通道共享电极(anode或cathode) → 空间相邻。
    这是TimeFilter模型所需的图结构。

    Returns:
        adj: (22, 22) 邻接矩阵，对称，对角线为1
    """
    n = N_TCP_CHANNELS
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            pair_i = set(TCP_BIPOLAR_PAIRS[i])
            pair_j = set(TCP_BIPOLAR_PAIRS[j])
            if pair_i & pair_j:  # 有共享电极 → 相邻
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    np.fill_diagonal(adj, 1.0)  # 自连接
    return adj


# ==============================================================================
# 脑区映射
# ==============================================================================

# TCP通道 → 脑区 (5脑区: frontal, temporal, central, parietal, occipital)
TCP_TO_REGION: Dict[str, str] = {
    # 左颞链
    'FP1-F7': 'frontal',   'F7-T3': 'temporal',
    'T3-T5': 'temporal',   'T5-O1': 'occipital',
    # 右颞链
    'FP2-F8': 'frontal',   'F8-T4': 'temporal',
    'T4-T6': 'temporal',   'T6-O2': 'occipital',
    # 中央链
    'A1-T3': 'temporal',   'T3-C3': 'central',
    'C3-CZ': 'central',    'CZ-C4': 'central',
    'C4-T4': 'central',    'T4-A2': 'temporal',
    # 左副矢状链
    'FP1-F3': 'frontal',   'F3-C3': 'central',
    'C3-P3': 'parietal',   'P3-O1': 'occipital',
    # 右副矢状链
    'FP2-F4': 'frontal',   'F4-C4': 'central',
    'C4-P4': 'parietal',   'P4-O2': 'occipital',
}

BRAIN_REGIONS = ['frontal', 'temporal', 'central', 'parietal', 'occipital']
REGION_TO_IDX = {r: i for i, r in enumerate(BRAIN_REGIONS)}
N_REGIONS = len(BRAIN_REGIONS)  # 5

# TCP通道 → 半球
TCP_TO_HEMISPHERE: Dict[str, str] = {
    'FP1-F7': 'L', 'F7-T3': 'L', 'T3-T5': 'L', 'T5-O1': 'L',
    'FP2-F8': 'R', 'F8-T4': 'R', 'T4-T6': 'R', 'T6-O2': 'R',
    'A1-T3': 'L', 'T3-C3': 'L',
    'C3-CZ': 'M', 'CZ-C4': 'M',
    'C4-T4': 'R', 'T4-A2': 'R',
    'FP1-F3': 'L', 'F3-C3': 'L', 'C3-P3': 'L', 'P3-O1': 'L',
    'FP2-F4': 'R', 'F4-C4': 'R', 'C4-P4': 'R', 'P4-O2': 'R',
}


# ==============================================================================
# 电极定义
# ==============================================================================

# TCP 22通道所需的全部单极电极（19个）
# 注意: FZ 和 PZ 不在TCP导联中，但属于标准10-20电极
TCP_REQUIRED_ELECTRODES = sorted(set(
    e for pair in TCP_BIPOLAR_PAIRS for e in pair
))
# = ['A1','A2','C3','C4','CZ','F3','F4','F7','F8','FP1','FP2',
#    'O1','O2','P3','P4','T3','T4','T5','T6']  共19个

# 提取电极列表（两种数据通用，21个 = 标准19 + A1 + A2）
# TUSZ的EDF文件中通道命名如 "EEG FP1-REF" 或 "EEG FP1-LE"
# 私有数据原始EDF通道如 "Fp1", "A1" 等
STANDARD_21_ELECTRODES = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'A1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'A2',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2',
]


# ==============================================================================
# 通道名称标准化
# ==============================================================================

# 通道名别名映射（处理各种命名变体）
CHANNEL_ALIAS_MAP = {
    # 10-10 → 10-20 映射
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    # 大小写变体（upper后的结果 → 标准名）
    'SPH-R': 'SPHR', 'SPH-L': 'SPHL',
    'SP1': 'SPHL', 'SPH1': 'SPHL', 'SP-L': 'SPHL',
    'SP2': 'SPHR', 'SPH2': 'SPHR', 'SP-R': 'SPHR',
    'SPH_L': 'SPHL', 'SPH_R': 'SPHR',
}

# EDF文件中常见的通道名 → 标准名
EDF_CHANNEL_MAP = {}
for _elec in STANDARD_21_ELECTRODES:
    EDF_CHANNEL_MAP[f'EEG {_elec}-REF'] = _elec
    EDF_CHANNEL_MAP[f'EEG {_elec}-LE']  = _elec
    EDF_CHANNEL_MAP[f'EEG {_elec}-AR']  = _elec
    EDF_CHANNEL_MAP[f'EEG {_elec}-AVG'] = _elec
# 额外的SPH通道（TUSZ中不常见，但留作兼容）
for _suf in ['REF', 'LE', 'AR', 'AVG']:
    EDF_CHANNEL_MAP[f'EEG SPHL-{_suf}'] = 'SPHL'
    EDF_CHANNEL_MAP[f'EEG SPHR-{_suf}'] = 'SPHR'
    EDF_CHANNEL_MAP[f'EEG SP1-{_suf}']  = 'SPHL'
    EDF_CHANNEL_MAP[f'EEG SP2-{_suf}']  = 'SPHR'


def normalize_channel_name(name: str) -> str:
    """
    标准化通道名称

    处理顺序:
    1. 查EDF格式映射表 ("EEG FP1-REF" → "FP1")
    2. 去除EDF前缀/后缀 → 尝试别名映射
    3. 直接upper作为结果

    Examples:
        "EEG FP1-REF" → "FP1"
        "Fp1"         → "FP1"
        "Sph-R"       → "SPHR"
        "T7"          → "T3"

    Args:
        name: 原始通道名

    Returns:
        标准化后的通道名（大写）
    """
    name = name.strip()
    name_upper = name.upper()

    # 查EDF格式映射
    if name_upper in EDF_CHANNEL_MAP:
        return EDF_CHANNEL_MAP[name_upper]

    # 去除EDF前缀/后缀
    stripped = name_upper
    for prefix in ['EEG ', 'EEG-']:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            for suffix in ['-REF', '-LE', '-AR', '-AVG']:
                if stripped.endswith(suffix):
                    stripped = stripped[:-len(suffix)]
            break

    # 查别名映射
    if stripped in CHANNEL_ALIAS_MAP:
        return CHANNEL_ALIAS_MAP[stripped]

    return stripped


# ==============================================================================
# TUSZ manifest列相关
# ==============================================================================

# TUSZ manifest中的18个per-channel标签列名
# 注意: TUSZ manifest没有中央链的6个通道列，但 onset_channels 字段包含全部22通道
TUSZ_MANIFEST_CHANNEL_COLUMNS = [
    'FP1_F7', 'F7_T3', 'T3_T5', 'T5_O1',
    'FP2_F8', 'F8_T4', 'T4_T6', 'T6_O2',
    'FP1_F3', 'F3_C3', 'C3_P3', 'P3_O1',
    'FP2_F4', 'F4_C4', 'C4_P4', 'P4_O2',
    'FZ_CZ', 'CZ_PZ',
]

# manifest列名 → TCP通道名 的映射
TUSZ_COL_TO_TCP = {
    'FP1_F7': 'FP1-F7', 'F7_T3': 'F7-T3', 'T3_T5': 'T3-T5', 'T5_O1': 'T5-O1',
    'FP2_F8': 'FP2-F8', 'F8_T4': 'F8-T4', 'T4_T6': 'T4-T6', 'T6_O2': 'T6-O2',
    'FP1_F3': 'FP1-F3', 'F3_C3': 'F3-C3', 'C3_P3': 'C3-P3', 'P3_O1': 'P3-O1',
    'FP2_F4': 'FP2-F4', 'F4_C4': 'F4-C4', 'C4_P4': 'C4-P4', 'P4_O2': 'P4-O2',
    # FZ_CZ 和 CZ_PZ 不在22 TCP通道中，作为额外信息保留
}


# ==============================================================================
# 私有数据manifest相关
# ==============================================================================

# 私有数据manifest中的21个per-electrode标签列名（小写）
PRIVATE_ELECTRODE_LABEL_COLUMNS = [
    'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
    't3', 'c3', 'cz', 'c4', 't4',
    't5', 'p3', 'pz', 'p4', 't6',
    'o1', 'o2', 'sph-l', 'sph-r'
]

# 私有数据manifest中的5个lateralized脑区列
PRIVATE_REGION_COLUMNS = [
    'left_frontal', 'left_temporal', 'parietal', 'right_frontal', 'right_temporal'
]

# lateralized脑区 → 统一5脑区映射
PRIVATE_REGION_TO_UNIFIED = {
    'left_frontal': 'frontal',
    'right_frontal': 'frontal',
    'left_temporal': 'temporal',
    'right_temporal': 'temporal',
    'parietal': 'parietal',
}


# ==============================================================================
# 预处理参数配置
# ==============================================================================

@dataclass
class PreprocessConfig:
    """数据预处理配置"""

    # ---- 数据类型 ----
    data_type: str = 'tusz'  # 'tusz' / 'private'

    # ---- 采样率 ----
    target_fs: float = 200.0

    # ---- 带通滤波 ----
    highpass_fc: float = 3.0
    lowpass_fc: float = 45.0
    filter_order: int = 4

    # ---- 幅值裁剪 ----
    clip_n_std: float = 1.0  # ±1个标准差

    # ---- 窗口参数 ----
    window_len: float = 12.0     # 秒 (12s × 200Hz = 2400采样点)
    window_overlap: float = 0.5  # 50%重叠

    # ---- 发作提取参数 ----
    pre_seizure_buffer: float = 5.0    # 发作前缓冲(秒)
    post_seizure_buffer: float = 5.0   # 发作后缓冲(秒)
    min_seizure_duration: float = 5.0  # 最小发作持续时间(秒)

    # ---- 基线参数 ----
    baseline_duration: float = 30.0     # 默认基线时长(秒)
    include_baseline: bool = True       # 是否包含基线样本

    # ---- 标准化 ----
    normalize_method: str = 'baseline'  # 'baseline' / 'zscore'

    # ---- TUSZ数据路径 ----
    tusz_data_root: str = r'F:\dataset\TUSZ\v2.0.3\edf'
    tusz_manifest: str = ''

    # ---- 私有数据路径 ----
    private_data_roots: List[str] = field(default_factory=lambda: [
        r"E:\DataSet\EEG\EEG dataset_SUAT",
    ])
    private_manifest: str = ''
    private_file_format: str = 'set'  # 'set' / 'edf'

    # ---- 输出 ----
    output_root: str = r'F:\process_dataset'   # 预处理根目录
    output_dir: str = ''   # 具体子目录 (自动设置)

    # ---- 统一 Manifest ----
    combined_manifest: str = ''  # combined_manifest.csv 路径

    def __post_init__(self):
        if not self.tusz_manifest:
            self.tusz_manifest = (
                r'E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\TUSZ\tusz_manifest.csv'
            )
        if not self.private_manifest:
            self.private_manifest = (
                r'E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\DeepSOZ\bipolar_manifest_expanded.csv'
            )
        if not self.output_dir:
            # 默认保存到 F:\process_dataset\{data_type}
            self.output_dir = str(Path(self.output_root) / self.data_type)

    @property
    def meta_dir(self) -> str:
        """全局元数据目录"""
        return str(Path(self.output_root) / 'meta')


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_channel_to_region_labels(channel_labels: np.ndarray) -> np.ndarray:
    """
    从22通道SOZ标签 → 5脑区SOZ标签

    Args:
        channel_labels: (22,) per-channel binary labels

    Returns:
        region_labels: (5,) per-region binary labels (OR逻辑)
    """
    region_labels = np.zeros(N_REGIONS, dtype=np.float32)
    for i, ch_name in enumerate(TCP_CHANNEL_NAMES):
        if channel_labels[i] > 0:
            region = TCP_TO_REGION.get(ch_name)
            if region and region in REGION_TO_IDX:
                region_labels[REGION_TO_IDX[region]] = 1.0
    return region_labels


# ==============================================================================
# 自测
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TCP 22-channel Bipolar Montage (TUSZ Official Order)")
    print("=" * 60)

    print(f"\nTCP Channels ({N_TCP_CHANNELS}):")
    for i, (name, pair) in enumerate(zip(TCP_CHANNEL_NAMES, TCP_BIPOLAR_PAIRS)):
        region = TCP_TO_REGION.get(name, '?')
        hemi = TCP_TO_HEMISPHERE.get(name, '?')
        chain = [k for k, v in TCP_CHAINS.items() if i in v][0]
        print(f"  {i:2d}: {name:10s}  chain={chain:22s}  region={region:10s}  hemi={hemi}")

    print(f"\nRequired electrodes ({len(TCP_REQUIRED_ELECTRODES)}): {TCP_REQUIRED_ELECTRODES}")
    print(f"Standard 21 electrodes: {STANDARD_21_ELECTRODES}")
    print(f"Brain Regions ({N_REGIONS}): {BRAIN_REGIONS}")

    adj = build_tcp_adjacency_matrix()
    n_edges = int((adj.sum() - N_TCP_CHANNELS) / 2)
    print(f"\nAdjacency Matrix: shape={adj.shape}, edges={n_edges}, "
          f"density={n_edges / (N_TCP_CHANNELS * (N_TCP_CHANNELS - 1) / 2):.3f}")

    print("\n--- Channel name normalization test ---")
    test_names = ['EEG FP1-REF', 'Fp1', 'Sph-R', 'T7', 'A1', 'EEG CZ-LE', 'Fz', 'Cz']
    for n in test_names:
        print(f"  '{n}' → '{normalize_channel_name(n)}'")

    print(f"\nDefault config:")
    cfg = PreprocessConfig()
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
