#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for strict binary seizure-stage pretraining on TUSZ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import logging
import csv

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover - runtime fallback for environments without torch
    torch = None
    Dataset = object
    _HAS_TORCH = False

try:
    from ..data_preprocess.eeg_pipeline import EEGPipeline, PipelineConfig
    _HAS_PIPELINE = True
except Exception:
    try:
        from data_preprocess.eeg_pipeline import EEGPipeline, PipelineConfig
        _HAS_PIPELINE = True
    except Exception:
        EEGPipeline = None
        PipelineConfig = object
        _HAS_PIPELINE = False


log = logging.getLogger(__name__)

NON_SEIZURE_LABEL = 0
SEIZURE_LABEL = 1
STAGE_IGNORE_INDEX = -100
LOAD_STATUS_OK = 'ok'
LOAD_STATUS_WINDOW_NONE = 'window_none'
LOAD_STATUS_BAD_WINDOW = 'bad_window'
LOAD_STATUS_INSUFFICIENT_CHANNELS = 'insufficient_channels'
LOAD_STATUS_NO_VALID_PATCH = 'no_valid_patch'
LOAD_STATUS_EXCEPTION = 'exception'
_STAGE_MANIFEST_COLS = [
    'source',
    'split',
    'patient_id',
    'edf_path',
    'duration',
    'sz_start',
    'sz_end',
]


@dataclass(frozen=True)
class StageSample:
    """Metadata describing one pretraining window."""

    row_idx: int
    source: str
    split: str
    patient_id: str
    edf_path: str
    duration_sec: float
    seizure_start_sec: float
    seizure_end_sec: float
    center_sec: float
    role: str


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def inspect_stage_annotation_support(
    manifest_path: str,
    tusz_data_root: str = '',
    source_filter: str = 'tusz',
    max_files: int = 128,
) -> Dict[str, object]:
    """
    Inspect whether TUSZ annotations support stage classification.

    Strict mode intentionally reports only binary seizure/non-seizure support.
    """
    rows: List[Dict[str, object]] = []
    normalized_filter = str(source_filter).strip().lower()
    with open(manifest_path, 'r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_source = str(row.get('source', '')).strip().lower()
            if normalized_filter not in ('all', 'both') and row_source != normalized_filter:
                continue
            duration = _safe_float(row.get('duration'), default=float('nan'))
            sz_start = _safe_float(row.get('sz_start'), default=float('nan'))
            sz_end = _safe_float(row.get('sz_end'), default=float('nan'))
            rows.append(
                {
                    'source': row_source,
                    'patient_id': row.get('patient_id', ''),
                    'edf_path': row.get('edf_path', ''),
                    'duration': duration,
                    'sz_start': sz_start,
                    'sz_end': sz_end,
                }
            )

    valid_rows = [
        row for row in rows
        if np.isfinite(row['duration']) and np.isfinite(row['sz_start']) and np.isfinite(row['sz_end'])
        and row['sz_end'] > row['sz_start']
    ]
    valid_patients = {str(row['patient_id']) for row in valid_rows}

    report: Dict[str, object] = {
        'task_mode': 'binary',
        'supported_classes': ['non_seizure', 'seizure'],
        'supports_preictal': False,
        'supports_postictal': False,
        'supports_4_class': False,
        'n_rows': int(len(rows)),
        'n_valid_events': int(len(valid_rows)),
        'n_unique_patients': int(len(valid_patients)),
        'positive_duration_hours': float(
            sum(float(row['sz_end']) - float(row['sz_start']) for row in valid_rows) / 3600.0
        ) if valid_rows else 0.0,
        'raw_annotation_counts': {},
    }

    if tusz_data_root:
        root = Path(tusz_data_root)
        raw_counts = {'csv': 0, 'csv_bi': 0, 'missing': 0}
        for row in valid_rows[:max_files]:
            edf_path = root / str(row['edf_path'])
            has_csv = edf_path.with_suffix('.csv').exists()
            has_csv_bi = edf_path.with_name(edf_path.stem + '.csv_bi').exists()
            if has_csv:
                raw_counts['csv'] += 1
            if has_csv_bi:
                raw_counts['csv_bi'] += 1
            if not has_csv and not has_csv_bi:
                raw_counts['missing'] += 1
        report['raw_annotation_counts'] = raw_counts

    return report


def assign_patch_binary_labels(
    seizure_start_sec: float,
    seizure_end_sec: float,
    window_start_sec: float,
    file_duration_sec: float,
    n_patches: int,
    patch_len: int,
    fs: float,
    ignore_index: int = STAGE_IGNORE_INDEX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign binary labels to each patch in a fixed-length window."""
    labels = np.full(n_patches, ignore_index, dtype=np.int64)
    valid_mask = np.zeros(n_patches, dtype=np.bool_)
    file_duration_sec = max(_safe_float(file_duration_sec, 0.0), 0.0)
    seizure_start_sec = np.clip(_safe_float(seizure_start_sec, 0.0), 0.0, file_duration_sec)
    seizure_end_sec = np.clip(_safe_float(seizure_end_sec, 0.0), 0.0, file_duration_sec)
    patch_sec = float(patch_len) / float(fs)

    for patch_idx in range(n_patches):
        patch_start = window_start_sec + patch_idx * patch_sec
        patch_end = patch_start + patch_sec
        is_valid = patch_start >= 0.0 and patch_end <= file_duration_sec
        if not is_valid:
            continue

        valid_mask[patch_idx] = True
        overlap = min(patch_end, seizure_end_sec) - max(patch_start, seizure_start_sec)
        labels[patch_idx] = SEIZURE_LABEL if overlap > 0.0 else NON_SEIZURE_LABEL

    return labels, valid_mask


def stage_collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Collate function for stage-pretraining windows."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for stage_collate_fn")
    return {
        'x': torch.stack([item['x'] for item in batch]),
        'channel_mask': torch.stack([item['channel_mask'] for item in batch]),
        'stage_labels': torch.stack([item['stage_labels'] for item in batch]),
        'stage_valid_mask': torch.stack([item['stage_valid_mask'] for item in batch]),
        'patient_id': [item['patient_id'] for item in batch],
        'edf_path': [item['edf_path'] for item in batch],
        'sample_role': [item['sample_role'] for item in batch],
        'load_status': [item['load_status'] for item in batch],
        'stage_valid_count': torch.tensor(
            [item['stage_valid_count'] for item in batch],
            dtype=torch.long,
        ),
        'channel_valid_count': torch.tensor(
            [item['channel_valid_count'] for item in batch],
            dtype=torch.long,
        ),
        'window_start_sec': torch.tensor([item['window_start_sec'] for item in batch], dtype=torch.float32),
        'sample_center_sec': torch.tensor([item['sample_center_sec'] for item in batch], dtype=torch.float32),
        'seizure_start_sec': torch.tensor([item['seizure_start_sec'] for item in batch], dtype=torch.float32),
        'seizure_end_sec': torch.tensor([item['seizure_end_sec'] for item in batch], dtype=torch.float32),
    }


class EEGStagePretrainDataset(Dataset):
    """Generate fixed-length EEG windows with binary patch labels."""

    def __init__(
        self,
        manifest_path: str,
        tusz_data_root: str,
        pipeline_cfg: PipelineConfig,
        source_filter: str = 'tusz',
        split_filter: Optional[Sequence[str]] = None,
        roles: Sequence[str] = ('onset', 'mid', 'offset'),
        ignore_index: int = STAGE_IGNORE_INDEX,
        center_jitter_sec: float = 0.0,
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for EEGStagePretrainDataset")
        if not _HAS_PIPELINE:
            raise ImportError("EEGPipeline dependencies are required for EEGStagePretrainDataset")
        self.manifest_path = manifest_path
        self.tusz_data_root = tusz_data_root
        self.pipeline = EEGPipeline(pipeline_cfg)
        self.roles = tuple(roles)
        self.ignore_index = ignore_index
        self.center_jitter_sec = max(float(center_jitter_sec), 0.0)
        self.df = self._load_manifest(manifest_path, source_filter, split_filter)
        self.samples = self._build_samples(self.df)
        self.zero_x = torch.zeros(
            22,
            self.pipeline.cfg.n_patches * self.pipeline.cfg.patch_len,
            dtype=torch.float32,
        )
        self.zero_channel_mask = torch.zeros(22, dtype=torch.float32)
        self.zero_stage_labels = torch.full(
            (self.pipeline.cfg.n_patches,),
            fill_value=ignore_index,
            dtype=torch.long,
        )
        self.zero_stage_valid_mask = torch.zeros(self.pipeline.cfg.n_patches, dtype=torch.bool)

        log.info(
            "EEGStagePretrainDataset: %d rows -> %d windows (source=%s, split=%s)",
            len(self.df),
            len(self.samples),
            source_filter,
            list(split_filter) if split_filter else None,
        )

    @staticmethod
    def _load_manifest(
        manifest_path: str,
        source_filter: str,
        split_filter: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        df = pd.read_csv(
            manifest_path,
            usecols=lambda col: col in _STAGE_MANIFEST_COLS,
            engine='python',
        )
        if 'source' in df.columns:
            df['source'] = df['source'].astype(str).str.strip().str.lower()
        if 'split' in df.columns:
            df['split'] = df['split'].astype(str).str.strip().str.lower()
        if source_filter not in ('all', 'both'):
            df = df[df['source'] == str(source_filter).strip().lower()]
        if split_filter:
            normalized_splits = [str(split).strip().lower() for split in split_filter]
            df = df[df['split'].isin(normalized_splits)]

        duration = pd.to_numeric(df.get('duration'), errors='coerce')
        sz_start = pd.to_numeric(df.get('sz_start'), errors='coerce')
        sz_end = pd.to_numeric(df.get('sz_end'), errors='coerce')
        valid = duration.notna() & sz_start.notna() & sz_end.notna() & (sz_end > sz_start)
        df = df.loc[valid].copy()
        df['duration'] = duration.loc[df.index]
        df['sz_start'] = sz_start.loc[df.index]
        df['sz_end'] = sz_end.loc[df.index]
        return df.reset_index(drop=True)

    def _build_samples(self, df: pd.DataFrame) -> List[StageSample]:
        window_sec = self.pipeline.cfg.pre_onset_sec + self.pipeline.cfg.post_onset_sec
        dedupe_tol = max(float(self.pipeline.cfg.patch_len) / float(self.pipeline.cfg.target_fs), 0.5)
        samples: List[StageSample] = []

        for row_idx, row in df.iterrows():
            duration_sec = _safe_float(row['duration'])
            seizure_start_sec = _safe_float(row['sz_start'])
            seizure_end_sec = _safe_float(row['sz_end'])
            if duration_sec <= 0.0 or seizure_end_sec <= seizure_start_sec:
                continue

            proposed: List[Tuple[str, float]] = []
            if 'onset' in self.roles:
                proposed.append(('onset', seizure_start_sec))
            if 'mid' in self.roles:
                proposed.append(('mid', 0.5 * (seizure_start_sec + seizure_end_sec)))
            if 'offset' in self.roles:
                proposed.append(('offset', seizure_end_sec))

            kept_centers: List[float] = []
            for role, center_sec in proposed:
                center_sec = np.clip(center_sec, 0.0, max(duration_sec, 0.0))
                if any(abs(center_sec - prev) < dedupe_tol for prev in kept_centers):
                    continue
                kept_centers.append(center_sec)
                samples.append(
                    StageSample(
                        row_idx=row_idx,
                        source=str(row.get('source', 'tusz')),
                        split=str(row.get('split', '')),
                        patient_id=str(row.get('patient_id', '')),
                        edf_path=str(row.get('edf_path', '')),
                        duration_sec=duration_sec,
                        seizure_start_sec=seizure_start_sec,
                        seizure_end_sec=seizure_end_sec,
                        center_sec=center_sec,
                        role=role,
                    )
                )

            if not proposed:
                # This should not happen in strict binary mode, but keep a fallback.
                center_sec = np.clip(seizure_start_sec + 0.5 * min(window_sec, seizure_end_sec - seizure_start_sec), 0.0, duration_sec)
                samples.append(
                    StageSample(
                        row_idx=row_idx,
                        source=str(row.get('source', 'tusz')),
                        split=str(row.get('split', '')),
                        patient_id=str(row.get('patient_id', '')),
                        edf_path=str(row.get('edf_path', '')),
                        duration_sec=duration_sec,
                        seizure_start_sec=seizure_start_sec,
                        seizure_end_sec=seizure_end_sec,
                        center_sec=center_sec,
                        role='fallback',
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_edf_path(self, sample: StageSample) -> str:
        if sample.source == 'tusz':
            return str(Path(self.tusz_data_root) / sample.edf_path)
        return sample.edf_path

    def _empty_item(self, sample: StageSample, window_start_sec: float, load_status: str) -> Dict[str, object]:
        return {
            'x': self.zero_x.clone(),
            'channel_mask': self.zero_channel_mask.clone(),
            'stage_labels': self.zero_stage_labels.clone(),
            'stage_valid_mask': self.zero_stage_valid_mask.clone(),
            'patient_id': sample.patient_id,
            'edf_path': sample.edf_path,
            'sample_role': sample.role,
            'load_status': load_status,
            'stage_valid_count': 0,
            'channel_valid_count': 0,
            'window_start_sec': float(window_start_sec),
            'sample_center_sec': float(sample.center_sec),
            'seizure_start_sec': float(sample.seizure_start_sec),
            'seizure_end_sec': float(sample.seizure_end_sec),
        }

    def _sample_center_sec(self, sample: StageSample) -> float:
        center_sec = float(sample.center_sec)
        if self.center_jitter_sec <= 0.0 or sample.role != 'onset':
            return center_sec

        patch_sec = float(self.pipeline.cfg.patch_len) / float(self.pipeline.cfg.target_fs)
        jitter_min = -(max(float(self.pipeline.cfg.post_onset_sec) - patch_sec, 0.0))
        jitter_max = max(float(self.pipeline.cfg.pre_onset_sec) - patch_sec, 0.0)
        lower = max(-self.center_jitter_sec, jitter_min, -center_sec)
        upper = min(
            self.center_jitter_sec,
            jitter_max,
            max(float(sample.duration_sec) - center_sec, 0.0),
        )
        if upper <= lower + 1e-6:
            return center_sec
        return float(center_sec + np.random.uniform(lower, upper))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        cfg = self.pipeline.cfg
        baseline_n = int(cfg.pre_onset_sec * cfg.target_fs)
        sample_center_sec = self._sample_center_sec(sample)
        window_start_sec = float(sample_center_sec - cfg.pre_onset_sec)

        try:
            data_21, fs = self.pipeline.load_edf(self._resolve_edf_path(sample), onset=sample_center_sec)
            window = self.pipeline.extract_window(data_21, fs, sample_center_sec)
            if window is None:
                item = self._empty_item(sample, window_start_sec, LOAD_STATUS_WINDOW_NONE)
                item['sample_center_sec'] = float(sample_center_sec)
                return item

            clipped = self.pipeline.clip_by_baseline(window, baseline_n)
            bipolar, channel_mask = self.pipeline.to_tcp_bipolar(clipped)
            if int(channel_mask.sum()) < cfg.min_valid_channels:
                item = self._empty_item(sample, window_start_sec, LOAD_STATUS_INSUFFICIENT_CHANNELS)
                item['sample_center_sec'] = float(sample_center_sec)
                return item
            if self.pipeline.is_bad_bipolar_window(bipolar, channel_mask, fs):
                item = self._empty_item(sample, window_start_sec, LOAD_STATUS_BAD_WINDOW)
                item['sample_center_sec'] = float(sample_center_sec)
                return item

            bipolar = self.pipeline.normalize_by_baseline(bipolar, baseline_n)
            stage_labels, valid_patch_mask = assign_patch_binary_labels(
                seizure_start_sec=sample.seizure_start_sec,
                seizure_end_sec=sample.seizure_end_sec,
                window_start_sec=window_start_sec,
                file_duration_sec=sample.duration_sec,
                n_patches=cfg.n_patches,
                patch_len=cfg.patch_len,
                fs=cfg.target_fs,
                ignore_index=self.ignore_index,
            )
            stage_valid_count = int(valid_patch_mask.sum())
            channel_valid_count = int(channel_mask.sum())
            if stage_valid_count <= 0:
                item = self._empty_item(sample, window_start_sec, LOAD_STATUS_NO_VALID_PATCH)
                item['sample_center_sec'] = float(sample_center_sec)
                return item

            return {
                'x': torch.from_numpy(bipolar.astype(np.float32)),
                'channel_mask': torch.from_numpy(channel_mask.astype(np.float32)),
                'stage_labels': torch.from_numpy(stage_labels).long(),
                'stage_valid_mask': torch.from_numpy(valid_patch_mask),
                'patient_id': sample.patient_id,
                'edf_path': sample.edf_path,
                'sample_role': sample.role,
                'load_status': LOAD_STATUS_OK,
                'stage_valid_count': stage_valid_count,
                'channel_valid_count': channel_valid_count,
                'window_start_sec': float(window_start_sec),
                'sample_center_sec': float(sample_center_sec),
                'seizure_start_sec': float(sample.seizure_start_sec),
                'seizure_end_sec': float(sample.seizure_end_sec),
            }
        except Exception as exc:
            log.warning("Stage window failed for %s (%s): %s", sample.edf_path, sample.role, exc)
            item = self._empty_item(sample, window_start_sec, LOAD_STATUS_EXCEPTION)
            item['sample_center_sec'] = float(sample_center_sec)
            return item


def summarize_stage_dataset(dataset: EEGStagePretrainDataset) -> Dict[str, object]:
    """Return simple metadata for logging."""
    role_counts: Dict[str, int] = {}
    for sample in dataset.samples:
        role_counts[sample.role] = role_counts.get(sample.role, 0) + 1
    return {
        'n_windows': len(dataset),
        'n_patients': int(dataset.df['patient_id'].nunique()) if len(dataset.df) else 0,
        'role_counts': role_counts,
    }
