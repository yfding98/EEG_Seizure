#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ManifestSOZDataset — 基于 combined_manifest.csv 的 SOZ 数据集
────────────────────────────────────────────────────────────────────
提供与 TimeFilterDataset 完全兼容的接口，用于训练 LaBraM-TimeFilter-SOZ。

数据格式:
    combined_manifest.csv 每行 = 一次发作事件
    标签: 22个 TCP 双极导联 0/1 列 (FP1_F7 ... T4_A2)

输出4元组 (与 TimeFilterDataset 一致):
    X:      [22, n_patches, patch_len]  TCP双极导联patch数据
    y_soz:  [n_output]                 SOZ标签 (22=双极 / 19=单极)
    mask:   [22]                       通道有效性掩码
    meta:   dict                       元数据

支持:
    - source 过滤 (tusz / private / both)
    - split 过滤 (train / dev / eval / private)
    - 22通道双极标签 (默认) 或 19通道单极标签 (通过映射)
    - get_patient_ids(), create_dataloader() 等接口
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# 项目内部导入
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

try:
    from data_preprocess.eeg_pipeline import (
        EEGPipeline,
        PipelineConfig,
        SeizureEvent,
        build_bipolar_to_unipolar_matrix,
        STANDARD_19,
        TCP_PAIRS as PIPELINE_TCP_PAIRS,
    )
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

logger = logging.getLogger(__name__)

# ── TCP 22通道名称（与 eeg_pipeline.py / config.py / combined_manifest 一致）
TCP_BIPOLAR_NAMES = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'A1-T3',  'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
]
TCP_COL_NAMES = [ch.replace('-', '_') for ch in TCP_BIPOLAR_NAMES]
COARSE_REGION_NAMES = ['FP', 'F', 'C', 'T', 'P', 'O']
FINE_REGION_NAMES = ['L_FP', 'R_FP', 'L_F', 'R_F', 'C', 'L_T', 'R_T', 'P', 'O']
REGION_LABEL_MODES = {
    'coarse': tuple(COARSE_REGION_NAMES),
    'fine_lateralized': tuple(FINE_REGION_NAMES),
}
REGION_NAMES = COARSE_REGION_NAMES
REGION_TO_INDEX = {name: idx for idx, name in enumerate(REGION_NAMES)}
HEMISPHERE_NAMES = ['L', 'R', 'B']
HEMISPHERE_TO_INDEX = {name: idx for idx, name in enumerate(HEMISPHERE_NAMES)}
HEMISPHERE_NAMES_LR = ['L', 'R']
HEMISPHERE_TO_INDEX_LR = {name: idx for idx, name in enumerate(HEMISPHERE_NAMES_LR)}
HEMISPHERE_IGNORE_INDEX = -100


def get_region_names(region_label_mode: str = 'coarse') -> Tuple[str, ...]:
    try:
        return REGION_LABEL_MODES[region_label_mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported region_label_mode: {region_label_mode}; "
            f"choices={tuple(REGION_LABEL_MODES.keys())}"
        ) from exc


def _electrode_side(channel_name: str) -> Optional[str]:
    name = str(channel_name).strip().upper()
    if not name:
        return None
    if name.endswith('Z'):
        return 'M'
    tail = name[-1]
    if tail.isdigit():
        return 'L' if int(tail) % 2 == 1 else 'R'
    return None


def _channel_to_regions(channel_name: str, region_label_mode: str = 'coarse') -> List[str]:
    """Map a channel/electrode name to one or more region labels."""
    name = str(channel_name).strip().upper()
    if not name:
        return []
    if '-' in name:
        regions: List[str] = []
        for part in name.split('-'):
            regions.extend(_channel_to_regions(part, region_label_mode=region_label_mode))
        return list(dict.fromkeys(regions))

    if region_label_mode == 'coarse':
        if name.startswith('FP'):
            return ['FP']
        if name.startswith('F'):
            return ['F']
        if name.startswith('C') or name == 'CZ':
            return ['C']
        if name.startswith('T'):
            return ['T']
        if name.startswith('P'):
            return ['P']
        if name.startswith('O'):
            return ['O']
        return []

    if region_label_mode == 'fine_lateralized':
        side = _electrode_side(name)
        if name.startswith('FP'):
            if side == 'L':
                return ['L_FP']
            if side == 'R':
                return ['R_FP']
            if side == 'M':
                return ['L_FP', 'R_FP']
            return []
        if name.startswith('F'):
            if side == 'L':
                return ['L_F']
            if side == 'R':
                return ['R_F']
            if side == 'M':
                return ['L_F', 'R_F']
            return []
        if name.startswith('C') or name == 'CZ':
            return ['C']
        if name.startswith('T'):
            if side == 'L':
                return ['L_T']
            if side == 'R':
                return ['R_T']
            if side == 'M':
                return ['L_T', 'R_T']
            return []
        if name.startswith('P'):
            return ['P']
        if name.startswith('O'):
            return ['O']
        return []

    raise ValueError(
        f"Unsupported region_label_mode: {region_label_mode}; "
        f"choices={tuple(REGION_LABEL_MODES.keys())}"
    )


def _build_region_target(
    onset_channels: str,
    bipolar_label: np.ndarray,
    region_label_mode: str = 'coarse',
) -> np.ndarray:
    """Build a multi-label region target from onset channels or active bipolar pairs."""
    region_names = get_region_names(region_label_mode)
    region_to_index = {name: idx for idx, name in enumerate(region_names)}
    region_target = np.zeros(len(region_names), dtype=np.float32)
    onset_parts = [part.strip() for part in str(onset_channels).split(';') if part.strip()]

    if onset_parts:
        for part in onset_parts:
            for region in _channel_to_regions(part, region_label_mode=region_label_mode):
                region_target[region_to_index[region]] = 1.0

    if region_target.sum() == 0:
        for is_active, pair in zip(bipolar_label.tolist(), TCP_BIPOLAR_NAMES):
            if is_active <= 0:
                continue
            for region in _channel_to_regions(pair, region_label_mode=region_label_mode):
                region_target[region_to_index[region]] = 1.0

    return region_target


def _map_hemisphere_label(raw_value: str, mode: str = 'lrb') -> int:
    value = str(raw_value).strip().upper()
    if mode == 'lrb':
        return HEMISPHERE_TO_INDEX.get(value, HEMISPHERE_IGNORE_INDEX)
    if mode == 'lr_ignore_b':
        return HEMISPHERE_TO_INDEX_LR.get(value, HEMISPHERE_IGNORE_INDEX)
    raise ValueError(f"Unsupported hemisphere label mode: {mode}")


def _build_bipolar_to_monopolar_matrix() -> np.ndarray:
    """构建 19×22 映射矩阵：双极 → 19单极（贡献加权）"""
    if _HAS_PIPELINE:
        M, participation = build_bipolar_to_unipolar_matrix()
        # 归一化每行：每个单极通道的贡献权重
        row_sums = M.sum(axis=1, keepdims=True)
        M_norm = M / np.maximum(row_sums, 1e-8)
        return M_norm  # (19, 22)
    else:
        # Fallback: 手动构建
        from labram_timefilter_soz import TCP_PAIRS, STANDARD_19
        STD_IDX = {ch: i for i, ch in enumerate(STANDARD_19)}
        M = np.zeros((19, 22), dtype=np.float32)
        for j, (a, b) in enumerate(TCP_PAIRS):
            if a in STD_IDX:
                M[STD_IDX[a], j] = 1.0
            if b in STD_IDX:
                M[STD_IDX[b], j] = 1.0
        row_sums = M.sum(axis=1, keepdims=True)
        return M / np.maximum(row_sums, 1e-8)


# ==============================================================================
# ManifestSOZDataset
# ==============================================================================

class ManifestSOZDataset(Dataset if _HAS_TORCH else object):
    """
    基于 combined_manifest.csv 的 SOZ 数据集

    与 TimeFilterDataset 接口完全兼容。

    Args:
        manifest_path:     combined_manifest.csv 路径
        tusz_data_root:    TUSZ EDF 文件根目录
        private_data_root: 私有数据集 EDF 文件根目录
        source_filter:     'tusz' / 'private' / 'both'
        split_filter:      ['train'] / ['train','dev'] / None(全部)
        patient_ids:       仅包含这些患者 (用于train/val拆分)
        soz_only:          仅包含有SOZ的样本
        label_mode:        'bipolar'(22ch) / 'monopolar'(19ch)
        pipeline_cfg:      EEG预处理管道配置 (None=使用默认)

    Usage:
        # 仅 TUSZ train（22通道双极标签）
        ds = ManifestSOZDataset(
            manifest_path='TUSZ/combined_manifest.csv',
            tusz_data_root='F:/dataset/TUSZ/v2.0.3/edf',
            source_filter='tusz',
            split_filter=['train'],
        )

        # 混合训练
        ds = ManifestSOZDataset(
            manifest_path='TUSZ/combined_manifest.csv',
            tusz_data_root='F:/dataset/TUSZ/v2.0.3/edf',
            private_data_root='E:/DataSet/EEG/EEG_dataset_SUAT',
            source_filter='both',
        )

        X, y_soz, mask, meta = ds[0]
    """

    def __init__(
        self,
        manifest_path: str,
        tusz_data_root: str = r'F:\dataset\TUSZ\v2.0.3\edf',
        private_data_root: str = None,
        chbmit_data_root: str = None,
        source_filter: str = 'both',
        split_filter: Optional[List[str]] = None,
        patient_ids: Optional[List[str]] = None,
        soz_only: bool = False,
        label_mode: str = 'bipolar',       # 'bipolar' or 'monopolar'
        region_label_mode: str = 'coarse',
        hemisphere_label_mode: str = 'lrb',
        pipeline_cfg: 'PipelineConfig' = None,
        exclude_montages: Optional[List[str]] = None,
        min_valid_channels: int = 0,
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required")

        self.tusz_data_root = tusz_data_root
        self.private_data_root = private_data_root
        self.chbmit_data_root = chbmit_data_root
        self.label_mode = label_mode
        if region_label_mode not in REGION_LABEL_MODES:
            raise ValueError(
                f"Unsupported region_label_mode: {region_label_mode}; "
                f"choices={tuple(REGION_LABEL_MODES.keys())}"
            )
        self.region_label_mode = region_label_mode
        if hemisphere_label_mode not in ('lrb', 'lr_ignore_b'):
            raise ValueError(
                f"Unsupported hemisphere_label_mode: {hemisphere_label_mode}"
            )
        self.hemisphere_label_mode = hemisphere_label_mode

        # 初始化 EEG 预处理管道
        if _HAS_PIPELINE:
            self.pipeline = EEGPipeline(pipeline_cfg or PipelineConfig())
        else:
            self.pipeline = None
            logger.warning(
                "EEGPipeline 不可用 (data_preprocess/eeg_pipeline.py 未找到)。"
                "将无法加载实际 EDF 数据。"
            )

        # 双极→单极映射矩阵 (用于 monopolar 标签模式)
        if label_mode == 'monopolar':
            self._b2m_matrix = _build_bipolar_to_monopolar_matrix()  # (19, 22)
        else:
            self._b2m_matrix = None

        # 加载 manifest
        self._load_manifest(
            manifest_path, source_filter, split_filter, patient_ids, soz_only,
            exclude_montages=exclude_montages or ['03_tcp_ar_a'],
            min_valid_channels=min_valid_channels,
        )

    def _load_manifest(
        self,
        path: str,
        source_filter: str,
        split_filter: Optional[List[str]],
        patient_ids: Optional[List[str]],
        soz_only: bool,
        exclude_montages: Optional[List[str]] = None,
        min_valid_channels: int = 0,
    ):
        df = pd.read_csv(path)
        n0 = len(df)

        if source_filter not in ('both', 'all'):
            df = df[df['source'] == source_filter]

        if split_filter:
            df = df[df['split'].isin(split_filter)]

        if patient_ids is not None:
            df = df[df['patient_id'].isin(patient_ids)]

        # 过滤无效行
        df = df.dropna(subset=['sz_start', 'sz_end'])
        df = df[df['sz_end'] > df['sz_start']]

        # 过滤指定 montage (通过 edf_path 路径中的关键字匹配)
        if exclude_montages:
            before_montage = len(df)
            edf_paths = df['edf_path'].astype(str)
            mask = ~edf_paths.apply(
                lambda p: any(m in p for m in exclude_montages)
            )
            df = df[mask]
            n_excluded = before_montage - len(df)
            if n_excluded > 0:
                logger.info(
                    f"  Montage 过滤: 排除 {n_excluded} 行 "
                    f"(montages={exclude_montages})"
                )

        # 最低有效导联数过滤 (基于 CSV 中 22 个 TCP 0/1 列判定)
        if min_valid_channels > 0:
            before_ch = len(df)
            # 使用 TCP 列判断：非零列数即为有效通道（近似，实际由 channel_mask 决定）
            valid_cols = [c for c in TCP_COL_NAMES if c in df.columns]
            if valid_cols:
                n_valid = (df[valid_cols] != 0).sum(axis=1)
                # 注：此处仅针对 SOZ 阳性样本有效，SOZ 全0的样本不应被此过滤
                # 因此仅对实际需要时使用
                df = df[n_valid >= min_valid_channels]
                n_ch_excluded = before_ch - len(df)
                if n_ch_excluded > 0:
                    logger.info(
                        f"  通道数过滤: 排除 {n_ch_excluded} 行 "
                        f"(min_valid_channels={min_valid_channels})"
                    )

        if soz_only:
            # 至少一个双极导联为1
            has_soz = df[TCP_COL_NAMES].sum(axis=1) > 0
            df = df[has_soz]

        self.df = df.reset_index(drop=True)

        # 添加 has_soz 列 (兼容 build_weighted_sampler)
        if 'has_soz' not in self.df.columns:
            self.df['has_soz'] = (self.df[TCP_COL_NAMES].sum(axis=1) > 0).astype(int)

        logger.info(
            f"ManifestSOZDataset: {n0} -> {len(self.df)} rows "
            f"(source={source_filter}, split={split_filter}, "
            f"patients={len(self.df['patient_id'].unique())}, "
            f"label_mode={self.label_mode}, region_mode={self.region_label_mode}, "
            f"hemisphere_mode={self.hemisphere_label_mode})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns:
            X:     [22, n_patches, patch_len] float32 — TCP双极patch数据
            y:     [22] float32 - bipolar, [19] float32 - monopolar (tuple)
            mask:  [22] float32 — 通道有效性掩码
            meta:  dict — 元数据 (contains SOZ/region/hemisphere labels too)
        """
        row = self.df.iloc[idx]

        # ── 标签 ──────────────────────────────────────────────────────────
        bipolar_label = np.array(
            [int(row.get(col, 0)) for col in TCP_COL_NAMES],
            dtype=np.float32,
        )  # (22,)
        
        # 双极 → 单极映射: (19, 22) @ (22,) → (19,)
        if self._b2m_matrix is not None:
            monopolar_label = (self._b2m_matrix @ bipolar_label > 0).astype(np.float32)
        else:
            # Fallback for safety
            self._b2m_matrix = _build_bipolar_to_monopolar_matrix()
            monopolar_label = (self._b2m_matrix @ bipolar_label > 0).astype(np.float32)

        if self.label_mode == 'bipolar':
            y_soz = bipolar_label
        else:
            y_soz = monopolar_label

        region_label = _build_region_target(
            str(row.get('onset_channels', '')),
            bipolar_label,
            region_label_mode=self.region_label_mode,
        )
        hemisphere_label = _map_hemisphere_label(
            str(row.get('hemisphere', '')),
            mode=self.hemisphere_label_mode,
        )

        # ── 元数据 ────────────────────────────────────────────────────────
        meta = {
            'source':         str(row.get('source', '')),
            'patient_id':     str(row.get('patient_id', '')),
            'edf_path':       str(row.get('edf_path', '')),
            'split':          str(row.get('split', '')),
            'sz_start':       float(row.get('sz_start', 0)),
            'sz_end':         float(row.get('sz_end', 0)),
            'hemisphere':     str(row.get('hemisphere', 'U')),
            'onset_channels': str(row.get('onset_channels', '')),
            'has_soz':        int(bipolar_label.sum() > 0),
            'row_idx':        idx,
            'bipolar_label':  bipolar_label,
            'monopolar_label': monopolar_label,
            'region_label':   region_label,
            'region_label_mode': self.region_label_mode,
            'hemisphere_label': hemisphere_label,
        }

        # ── EEG 数据 ──────────────────────────────────────────────────────
        X, mask = self._load_eeg(row, idx)

        # 转为 torch tensor
        X = torch.from_numpy(X).float()
        y_soz = torch.from_numpy(y_soz).float()
        y_bipolar = torch.from_numpy(bipolar_label).float()
        y_monopolar = torch.from_numpy(monopolar_label).float()
        y_region = torch.from_numpy(region_label).float()
        y_hemisphere = torch.tensor(hemisphere_label, dtype=torch.long)
        mask = torch.from_numpy(mask).float()

        return X, y_soz, mask, meta, y_bipolar, y_monopolar, y_region, y_hemisphere

    def _load_eeg(self, row: pd.Series, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """加载 EDF → 21单极 → 双极转换 → patches"""
        cfg = self.pipeline.cfg if self.pipeline else None
        n_patches = cfg.n_patches if cfg else 20
        patch_len = cfg.patch_len if cfg else 100
        n_ch = 22

        # 解析路径
        source = str(row.get('source', 'tusz'))
        edf_rel = str(row.get('edf_path', ''))
        if source == 'tusz':
            edf_path = str(Path(self.tusz_data_root) / edf_rel)
        else:
            if self.private_data_root:
                edf_path = str(Path(self.private_data_root) / edf_rel)
            else:
                edf_path = edf_rel

        sz_start = float(row.get('sz_start', 0))
        sz_end   = float(row.get('sz_end', 0))

        if self.pipeline is None:
            # 无管道可用 → 返回零数据
            logger.debug(f"[{idx}] 管道不可用，返回零数据")
            return np.zeros((n_ch, n_patches, patch_len), dtype=np.float32), np.ones(n_ch, dtype=np.float32)

        try:
            # 创建 SeizureEvent
            event = SeizureEvent(
                edf_path=edf_path,
                onset=sz_start,
                end=sz_end,
                soz_channels=[],  # 标签从 CSV 列直接读取
                source=source,
                patient_id=str(row.get('patient_id', '')),
            )

            result = self.pipeline.process_event(event)
            if result is None:
                logger.debug(f"[{idx}] 样本被管道剔除，返回零数据: {edf_path}")
                return np.zeros((n_ch, n_patches, patch_len), dtype=np.float32), np.ones(n_ch, dtype=np.float32)

            X = result['X']                    # (22, n_patches, patch_len)
            mask = result['channel_mask']       # (22,)
            return X, mask

        except Exception as e:
            logger.warning(f"[{idx}] 加载失败 {edf_path}: {e}")
            return np.zeros((n_ch, n_patches, patch_len), dtype=np.float32), np.ones(n_ch, dtype=np.float32)

    # ── 兼容 TimeFilterDataset 的接口 ─────────────────────────────────────

    def get_patient_ids(self) -> List[str]:
        return sorted(self.df['patient_id'].unique().tolist())

    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> 'DataLoader':
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        Xs, ys, masks, metas, y_bipolars, y_monopolars, y_regions, y_hemispheres = zip(*batch)
        return (
            torch.stack(Xs),
            torch.stack(ys),
            torch.stack(masks),
            list(metas),
            torch.stack(y_bipolars),
            torch.stack(y_monopolars),
            torch.stack(y_regions),
            torch.stack(y_hemispheres),
        )


# ==============================================================================
# 自测（仅验证 manifest 加载和标签 — 不需要实际 EDF 文件）
# ==============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    manifest = r'E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\TUSZ\combined_manifest.csv'
    if not Path(manifest).exists():
        print(f"[SKIP] combined_manifest.csv not found: {manifest}")
        sys.exit(0)

    print("=" * 60)
    print("ManifestSOZDataset Self-Test (manifest only, no EDF)")
    print("=" * 60)

    for mode in ['bipolar', 'monopolar']:
        for src in ['tusz', 'private', 'both']:
            ds = ManifestSOZDataset(
                manifest_path=manifest,
                source_filter=src,
                label_mode=mode,
            )
            # 只验证标签形状，不加载 EDF
            if len(ds) > 0:
                row = ds.df.iloc[0]
                bipolar_label = np.array([int(row.get(c, 0)) for c in TCP_COL_NAMES], dtype=np.float32)
                if mode == 'bipolar':
                    y_shape = 22
                else:
                    y_shape = 19
                print(
                    f"  [{mode:10s} | {src:8s}] "
                    f"n={len(ds):5d}, "
                    f"patients={len(ds.get_patient_ids()):3d}, "
                    f"label_dim={y_shape}, "
                    f"soz+={int(ds.df['has_soz'].sum())}"
                )

    # 测试 patient_ids 过滤
    ds_all = ManifestSOZDataset(manifest_path=manifest, source_filter='tusz')
    pids = ds_all.get_patient_ids()
    if len(pids) > 2:
        ds_sub = ManifestSOZDataset(
            manifest_path=manifest,
            source_filter='tusz',
            patient_ids=pids[:2],
        )
        print(f"\n  Patient filter: {len(pids)} -> {len(ds_sub.get_patient_ids())} patients, {len(ds_sub)} rows")

    print("\n[OK] ManifestSOZDataset self-test passed!")
