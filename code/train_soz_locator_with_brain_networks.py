#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_soz_locator_with_brain_networks.py

端到端 SOZ 定位训练脚本，集成脑网络特征。

流程:
  1. 数据加载 (ManifestSOZDataset)
  2. (可选) 对比学习预训练
  3. 三阶段微调 (冻结骨干 -> 解冻TimeFilter -> 全模型)
  4. 测试评估 + 可解释性报告

支持: DDP / AMP / 断点续训 / TensorBoard
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
    WeightedRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

SUPPORTED_BRAIN_NETWORK_FEATURES: Tuple[str, ...] = ('gc', 'te', 'aec', 'wpli')

# ── project path ──
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# ── imports ──
try:
    from models.integration_model import (
        TimeFilter_LaBraM_BrainNetwork_Integration, IntegrationConfig,
    )
    from models.contrastive_pretrainer import (
        BrainNetworkContrastivePretrainer, PretrainConfig,
    )
    from models.brain_network_extractor import MultiScaleBrainNetworkExtractor
    from models.dynamic_network_evolution import DynamicNetworkEvolutionModel
    from models.manifest_dataset import (
        ManifestSOZDataset,
        STANDARD_19,
        TCP_BIPOLAR_NAMES,
        TCP_COL_NAMES,
        _build_bipolar_to_monopolar_matrix,
        get_region_names,
    )
    from models.region_confusion import save_region_confusion_report
    from tasks.stage_detection import (
        EEGStagePretrainDataset,
        NON_SEIZURE_LABEL,
        SEIZURE_LABEL,
        assign_patch_binary_labels,
        inspect_stage_annotation_support,
        stage_collate_fn,
        summarize_stage_dataset,
    )
    from tasks.stage_seizure_metrics import (
        compute_deepsoz_stage_metrics,
        run_detailed_evaluation as run_deepsoz_detailed_evaluation,
    )
    from tasks.soz_localization_metrics import (
        compute_deepsoz_soz_metrics,
        compute_deepsoz_soz_metrics_mc,
        run_detailed_soz_evaluation,
    )
except ImportError:
    from .integration_model import (
        TimeFilter_LaBraM_BrainNetwork_Integration, IntegrationConfig,
    )
    from .contrastive_pretrainer import (
        BrainNetworkContrastivePretrainer, PretrainConfig,
    )
    from .brain_network_extractor import MultiScaleBrainNetworkExtractor
    from .dynamic_network_evolution import DynamicNetworkEvolutionModel
    from .manifest_dataset import (
        ManifestSOZDataset,
        STANDARD_19,
        TCP_BIPOLAR_NAMES,
        TCP_COL_NAMES,
        _build_bipolar_to_monopolar_matrix,
        get_region_names,
    )
    from .region_confusion import save_region_confusion_report
    from ..tasks.stage_detection import (
        EEGStagePretrainDataset,
        NON_SEIZURE_LABEL,
        SEIZURE_LABEL,
        assign_patch_binary_labels,
        inspect_stage_annotation_support,
        stage_collate_fn,
        summarize_stage_dataset,
    )
    from ..tasks.stage_seizure_metrics import (
        compute_deepsoz_stage_metrics,
        run_detailed_evaluation as run_deepsoz_detailed_evaluation,
    )
    from ..tasks.soz_localization_metrics import (
        compute_deepsoz_soz_metrics,
        compute_deepsoz_soz_metrics_mc,
        run_detailed_soz_evaluation,
    )

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

log = logging.getLogger('train_bn')
MONOPOLAR_NAMES: Tuple[str, ...] = tuple(str(ch) for ch in STANDARD_19)
MIRROR_ELECTRODE_MAP: Dict[str, str] = {
    'FP1': 'FP2',
    'FP2': 'FP1',
    'F3': 'F4',
    'F4': 'F3',
    'C3': 'C4',
    'C4': 'C3',
    'P3': 'P4',
    'P4': 'P3',
    'O1': 'O2',
    'O2': 'O1',
    'F7': 'F8',
    'F8': 'F7',
    'T3': 'T4',
    'T4': 'T3',
    'T5': 'T6',
    'T6': 'T5',
    'A1': 'A2',
    'A2': 'A1',
    'FZ': 'FZ',
    'CZ': 'CZ',
    'PZ': 'PZ',
}


# =====================================================================
# Helpers
# =====================================================================

def setup_logging(output_dir: Path, rank: int = 0):
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.append(logging.FileHandler(output_dir / 'train.log'))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def setup_ddp():
    """Initialise DDP if launched via torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def _summarize_manifest_subset(manifest_ds: ManifestSOZDataset) -> Dict[str, object]:
    df = manifest_ds.df
    if len(df) == 0:
        return {'rows': 0, 'patients': 0, 'sources': {}, 'hemisphere': {}}
    return {
        'rows': int(len(df)),
        'patients': int(df['patient_id'].nunique()),
        'sources': {str(k): int(v) for k, v in df['source'].value_counts().to_dict().items()},
        'hemisphere': {str(k): int(v) for k, v in df['hemisphere'].value_counts().to_dict().items()},
    }


def _format_subset_summary(name: str, summary: Dict[str, object]) -> str:
    return (
        f"{name}: rows={summary['rows']} patients={summary['patients']} "
        f"sources={summary['sources']} hemisphere={summary['hemisphere']}"
    )


def _build_signed_bipolar_mirror_permutation(
    channel_names: Tuple[str, ...],
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    name_to_index = {str(name): idx for idx, name in enumerate(channel_names)}
    mirror_index: List[int] = []
    mirror_sign: List[float] = []
    for name in channel_names:
        left, right = str(name).split('-', 1)
        mirrored = f"{MIRROR_ELECTRODE_MAP[left]}-{MIRROR_ELECTRODE_MAP[right]}"
        if mirrored in name_to_index:
            mirror_index.append(name_to_index[mirrored])
            mirror_sign.append(1.0)
            continue
        reversed_mirrored = f"{MIRROR_ELECTRODE_MAP[right]}-{MIRROR_ELECTRODE_MAP[left]}"
        if reversed_mirrored not in name_to_index:
            raise KeyError(f"Cannot build left-right mirror mapping for bipolar channel: {name}")
        mirror_index.append(name_to_index[reversed_mirrored])
        mirror_sign.append(-1.0)
    return tuple(mirror_index), tuple(mirror_sign)


def _build_monopolar_mirror_permutation(
    channel_names: Tuple[str, ...],
) -> Tuple[int, ...]:
    name_to_index = {str(name): idx for idx, name in enumerate(channel_names)}
    mirror_index: List[int] = []
    for name in channel_names:
        mirrored = MIRROR_ELECTRODE_MAP[str(name)]
        if mirrored not in name_to_index:
            raise KeyError(f"Cannot build left-right mirror mapping for monopolar channel: {name}")
        mirror_index.append(name_to_index[mirrored])
    return tuple(mirror_index)


BIPOLAR_MIRROR_INDEX, BIPOLAR_MIRROR_SIGN = _build_signed_bipolar_mirror_permutation(
    tuple(TCP_BIPOLAR_NAMES)
)
MONOPOLAR_MIRROR_INDEX = _build_monopolar_mirror_permutation(MONOPOLAR_NAMES)


def apply_lateral_mirror_augmentation(
    x: torch.Tensor,
    label: torch.Tensor,
    bipolar_label: torch.Tensor,
    monopolar_label: torch.Tensor,
    region_label: torch.Tensor,
    hemisphere_label: torch.Tensor,
    mirror_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mirror_prob <= 0.0 or x.size(0) == 0:
        return x, label, bipolar_label, monopolar_label, region_label, hemisphere_label

    eligible_mask = (hemisphere_label == 0) | (hemisphere_label == 1)
    random_mask = torch.rand(x.size(0), device=x.device) < float(mirror_prob)
    mirror_mask = eligible_mask & random_mask
    if not bool(mirror_mask.any()):
        return x, label, bipolar_label, monopolar_label, region_label, hemisphere_label

    x = x.clone()
    label = label.clone()
    bipolar_label = bipolar_label.clone()
    monopolar_label = monopolar_label.clone()
    hemisphere_label = hemisphere_label.clone()

    selected = torch.nonzero(mirror_mask, as_tuple=False).flatten()
    bipolar_index = torch.tensor(BIPOLAR_MIRROR_INDEX, device=x.device, dtype=torch.long)
    bipolar_sign = torch.tensor(BIPOLAR_MIRROR_SIGN, device=x.device, dtype=x.dtype).view(1, -1, 1)
    mono_index = torch.tensor(MONOPOLAR_MIRROR_INDEX, device=monopolar_label.device, dtype=torch.long)

    mirrored_x = x[selected].index_select(1, bipolar_index) * bipolar_sign
    mirrored_bipolar = bipolar_label[selected].index_select(1, bipolar_index)
    mirrored_monopolar = monopolar_label[selected].index_select(1, mono_index)

    x[selected] = mirrored_x
    bipolar_label[selected] = mirrored_bipolar
    monopolar_label[selected] = mirrored_monopolar

    if label.shape[1] == mirrored_bipolar.shape[1]:
        label[selected] = mirrored_bipolar
    elif label.shape[1] == mirrored_monopolar.shape[1]:
        label[selected] = mirrored_monopolar
    else:
        raise ValueError(
            f"Unsupported label dimension for mirror augmentation: {tuple(label.shape)}"
        )

    left_mask = hemisphere_label[selected] == 0
    right_mask = hemisphere_label[selected] == 1
    hemisphere_label[selected[left_mask]] = 1
    hemisphere_label[selected[right_mask]] = 0
    return x, label, bipolar_label, monopolar_label, region_label, hemisphere_label


def _materialize_manifest_metadata(dataset) -> Optional[Dict[str, object]]:
    if isinstance(dataset, Subset):
        meta = _materialize_manifest_metadata(dataset.dataset)
        if meta is None:
            return None
        df = meta['df'].iloc[list(dataset.indices)].reset_index(drop=True)
        return {'df': df, 'label_mode': meta['label_mode']}

    if isinstance(dataset, ConcatDataset):
        parts: List[pd.DataFrame] = []
        label_mode: Optional[str] = None
        for sub_dataset in dataset.datasets:
            meta = _materialize_manifest_metadata(sub_dataset)
            if meta is None:
                return None
            if label_mode is None:
                label_mode = str(meta['label_mode'])
            elif str(meta['label_mode']) != label_mode:
                raise ValueError(
                    "ConcatDataset contains inconsistent label modes: "
                    f"{label_mode} vs {meta['label_mode']}"
                )
            parts.append(meta['df'])
        if not parts:
            return None
        return {
            'df': pd.concat(parts, ignore_index=True),
            'label_mode': label_mode or 'monopolar',
        }

    if hasattr(dataset, 'ds') and isinstance(dataset.ds, ManifestSOZDataset):
        return {
            'df': dataset.ds.df.reset_index(drop=True).copy(),
            'label_mode': dataset.ds.label_mode,
        }

    if isinstance(dataset, ManifestSOZDataset):
        return {
            'df': dataset.df.reset_index(drop=True).copy(),
            'label_mode': dataset.label_mode,
        }

    return None


def _build_label_matrix(
    df: pd.DataFrame,
    label_mode: str,
) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray]:
    bipolar = df[TCP_COL_NAMES].fillna(0).to_numpy(dtype=np.float32, copy=True)
    if str(label_mode) == 'bipolar':
        return bipolar, tuple(TCP_BIPOLAR_NAMES), bipolar

    b2m = _build_bipolar_to_monopolar_matrix()  # (19, 22)
    monopolar = (bipolar @ b2m.T > 0).astype(np.float32)
    return monopolar, MONOPOLAR_NAMES, bipolar


def analyze_training_labels(dataset) -> Optional[Dict[str, object]]:
    meta = _materialize_manifest_metadata(dataset)
    if meta is None:
        return None

    df = meta['df']
    labels, channel_names, bipolar_labels = _build_label_matrix(df, str(meta['label_mode']))
    return {
        'df': df,
        'labels': labels,
        'bipolar_labels': bipolar_labels,
        'channel_names': channel_names,
        'label_mode': str(meta['label_mode']),
    }


def compute_pos_weight_from_analysis(
    analysis: Dict[str, object],
    device='cpu',
) -> torch.Tensor:
    labels = np.asarray(analysis['labels'], dtype=np.float32)
    if labels.size == 0:
        raise ValueError("Cannot compute pos_weight from an empty training set")

    pos_sum = torch.from_numpy(labels.sum(axis=0)).float()
    total = labels.shape[0]
    neg_sum = float(total) - pos_sum
    pw = (neg_sum / pos_sum.clamp(min=1.0)).float().clamp(max=50.0)

    pos_rate = pos_sum / max(float(total), 1.0)
    named_rates = ', '.join(
        f"{name}:{rate:.3f}"
        for name, rate in zip(analysis['channel_names'], pos_rate.tolist())
    )
    log.info(
        "pos_weight from manifest labels: min=%.1f max=%.1f mean=%.1f global_pos_rate=%.4f",
        float(pw.min().item()),
        float(pw.max().item()),
        float(pw.mean().item()),
        float(labels.mean()),
    )
    log.info("  per-channel pos_rate: %s", named_rates)
    return pw.to(device)


def build_private_channel_weight(
    analysis: Optional[Dict[str, object]],
    min_weight: float,
    max_weight: float,
    zero_positive_weight: float,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, object]]]:
    if analysis is None:
        return None, None

    df = analysis['df']
    sources = {str(s).strip().lower() for s in df['source'].tolist()}
    if sources != {'private'}:
        return None, None

    labels = np.asarray(analysis['labels'], dtype=np.float32)
    pos_counts = labels.sum(axis=0)
    weights = np.ones(labels.shape[1], dtype=np.float32)
    nonzero_mask = pos_counts > 0

    if np.any(nonzero_mask):
        reference = float(np.median(pos_counts[nonzero_mask]))
        scaled = np.sqrt(reference / np.maximum(pos_counts, 1.0))
        weights[nonzero_mask] = np.clip(
            scaled[nonzero_mask],
            min_weight,
            max_weight,
        )
    weights[~nonzero_mask] = float(zero_positive_weight)

    zero_positive_channels = [
        name
        for name, count in zip(analysis['channel_names'], pos_counts.tolist())
        if count <= 0
    ]
    rare_order = np.argsort(pos_counts)
    ranked = [
        f"{analysis['channel_names'][idx]}:{int(pos_counts[idx])}->{weights[idx]:.2f}"
        for idx in rare_order[: min(8, len(rare_order))]
    ]
    summary = {
        'zero_positive_channels': zero_positive_channels,
        'ranked_channel_weights': ranked,
    }
    return torch.tensor(weights, dtype=torch.float32, device=device), summary


def build_private_weighted_sampler(
    analysis: Optional[Dict[str, object]],
    patient_power: float,
    rare_channel_strength: float,
    rare_channel_max_boost: float,
    sample_weight_cap: float,
) -> Tuple[Optional[WeightedRandomSampler], Optional[Dict[str, object]]]:
    if analysis is None:
        return None, None

    df = analysis['df']
    sources = {str(s).strip().lower() for s in df['source'].tolist()}
    if sources != {'private'} or len(df) == 0:
        return None, None

    patient_ids = [str(pid) for pid in df['patient_id'].tolist()]
    patient_counts = Counter(patient_ids)
    patient_weights = np.asarray(
        [1.0 / max(patient_counts[pid], 1) for pid in patient_ids],
        dtype=np.float64,
    )
    patient_power = max(float(patient_power), 0.0)
    if patient_power != 1.0:
        patient_weights = np.power(patient_weights, patient_power)
    patient_weights = patient_weights / max(patient_weights.mean(), 1e-8)

    labels = np.asarray(analysis['labels'], dtype=np.float32)
    pos_counts = labels.sum(axis=0)
    sample_channel_weights = np.ones(len(labels), dtype=np.float64)
    nonzero_mask = pos_counts > 0
    if np.any(nonzero_mask):
        reference = float(np.median(pos_counts[nonzero_mask]))
        channel_boost = np.ones(labels.shape[1], dtype=np.float64)
        channel_boost[nonzero_mask] = np.clip(
            np.sqrt(reference / np.maximum(pos_counts[nonzero_mask], 1.0)),
            1.0,
            float(max(rare_channel_max_boost, 1.0)),
        )
        for i, row in enumerate(labels):
            positive = row > 0.5
            if np.any(positive):
                sample_channel_weights[i] = float(channel_boost[positive].mean())

    strength = float(np.clip(rare_channel_strength, 0.0, 1.0))
    sample_weights = patient_weights * (
        (1.0 - strength) + strength * sample_channel_weights
    )
    cap = float(max(sample_weight_cap, 1.0))
    sample_weights = np.clip(sample_weights, 1.0 / cap, cap)
    sample_weights = sample_weights / max(sample_weights.mean(), 1e-8)

    summary = {
        'patient_counts': dict(patient_counts),
        'weight_min': float(sample_weights.min()),
        'weight_max': float(sample_weights.max()),
        'weight_mean': float(sample_weights.mean()),
    }
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, summary


class EEGWindowAugmentor:
    """EEG signal-level data augmentation.

    Methods from Rommel et al. "Data augmentation for learning predictive
    models on EEG: a systematic comparison" and EEGConformer:

    - Gaussian noise addition
    - Random narrow band-stop filtering
    - Channel dropout (zeroing)
    - Smooth time mask (Hann-window temporal masking)          [NEW]
    - Per-channel amplitude scaling                             [NEW]
    - Frequency shift (spectral phase rotation)                 [NEW]
    - Temporal shift / jitter (circular shift)                  [NEW]
    """

    def __init__(
        self,
        fs: float,
        # --- existing ---
        gaussian_prob: float = 0.0,
        gaussian_std_scale: float = 0.01,
        bandstop_prob: float = 0.0,
        bandstop_min_freq: float = 45.0,
        bandstop_max_freq: float = 65.0,
        bandstop_width_hz: float = 2.0,
        channel_dropout_prob: float = 0.0,
        max_channel_drops: int = 1,
        # --- new: Rommel et al. ---
        time_mask_prob: float = 0.0,
        time_mask_max_ratio: float = 0.2,
        amplitude_scale_prob: float = 0.0,
        amplitude_scale_min: float = 0.8,
        amplitude_scale_max: float = 1.2,
        freq_shift_prob: float = 0.0,
        freq_shift_max_hz: float = 2.0,
        time_shift_prob: float = 0.0,
        time_shift_max_samples: int = 50,
    ):
        self.fs = float(fs)
        self.gaussian_prob = float(max(gaussian_prob, 0.0))
        self.gaussian_std_scale = float(max(gaussian_std_scale, 0.0))
        self.bandstop_prob = float(max(bandstop_prob, 0.0))
        self.bandstop_min_freq = float(max(bandstop_min_freq, 0.0))
        self.bandstop_max_freq = float(max(bandstop_max_freq, self.bandstop_min_freq))
        self.bandstop_width_hz = float(max(bandstop_width_hz, 0.1))
        self.channel_dropout_prob = float(max(channel_dropout_prob, 0.0))
        self.max_channel_drops = int(max(max_channel_drops, 0))
        # new
        self.time_mask_prob = float(max(time_mask_prob, 0.0))
        self.time_mask_max_ratio = float(np.clip(time_mask_max_ratio, 0.01, 0.5))
        self.amplitude_scale_prob = float(max(amplitude_scale_prob, 0.0))
        self.amplitude_scale_min = float(amplitude_scale_min)
        self.amplitude_scale_max = float(max(amplitude_scale_max, self.amplitude_scale_min))
        self.freq_shift_prob = float(max(freq_shift_prob, 0.0))
        self.freq_shift_max_hz = float(max(freq_shift_max_hz, 0.0))
        self.time_shift_prob = float(max(time_shift_prob, 0.0))
        self.time_shift_max_samples = int(max(time_shift_max_samples, 0))

    # ---- existing methods ----

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.gaussian_prob <= 0.0 or self.gaussian_std_scale <= 0.0:
            return x
        batch_mask = (torch.rand(x.size(0), device=x.device) < self.gaussian_prob).float()
        if batch_mask.sum() == 0:
            return x
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(x) * std * self.gaussian_std_scale
        return x + noise * batch_mask.view(-1, 1, 1)

    def _bandstop(self, x: torch.Tensor) -> torch.Tensor:
        if self.bandstop_prob <= 0.0:
            return x
        batch_mask = torch.rand(x.size(0), device=x.device) < self.bandstop_prob
        if not bool(batch_mask.any()):
            return x

        selected = torch.nonzero(batch_mask, as_tuple=False).flatten()
        if selected.numel() == 0:
            return x

        n_time = x.size(-1)
        freqs = torch.fft.rfftfreq(n_time, d=1.0 / self.fs).to(x.device)
        transformed = torch.fft.rfft(x[selected], dim=-1)
        half_width = self.bandstop_width_hz / 2.0
        for local_idx in range(selected.numel()):
            center = torch.empty(1, device=x.device).uniform_(
                self.bandstop_min_freq,
                self.bandstop_max_freq,
            ).item()
            keep_mask = ((freqs < center - half_width) | (freqs > center + half_width)).to(transformed.dtype)
            transformed[local_idx] = transformed[local_idx] * keep_mask.view(1, -1)
        x = x.clone()
        x[selected] = torch.fft.irfft(transformed, n=n_time, dim=-1)
        return x

    def _channel_dropout(
        self,
        x: torch.Tensor,
        bipolar_label: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.channel_dropout_prob <= 0.0 or self.max_channel_drops <= 0:
            return x

        x = x.clone()
        n_channels = x.size(1)
        for batch_idx in range(x.size(0)):
            if torch.rand(1, device=x.device).item() >= self.channel_dropout_prob:
                continue

            if bipolar_label is not None:
                positive = torch.nonzero(bipolar_label[batch_idx] > 0.5, as_tuple=False).flatten()
            else:
                positive = torch.empty(0, dtype=torch.long, device=x.device)

            if positive.numel() <= 1:
                candidates = torch.nonzero(
                    bipolar_label[batch_idx] <= 0.5 if bipolar_label is not None else torch.ones(n_channels, device=x.device),
                    as_tuple=False,
                ).flatten()
            else:
                candidates = torch.arange(n_channels, device=x.device)

            if candidates.numel() == 0:
                continue

            n_drop = min(
                int(torch.randint(1, self.max_channel_drops + 1, (1,), device=x.device).item()),
                int(candidates.numel()),
            )
            perm = torch.randperm(int(candidates.numel()), device=x.device)[:n_drop]
            x[batch_idx, candidates[perm], :] = 0.0
        return x

    # ---- new methods (Rommel et al.) ----

    def _smooth_time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a smooth Hann-window mask to a random temporal segment.

        Forces the model to rely on surrounding temporal context instead of
        memorising a fixed pattern at a particular time offset.
        """
        if self.time_mask_prob <= 0.0:
            return x
        batch_mask = torch.rand(x.size(0), device=x.device) < self.time_mask_prob
        if not bool(batch_mask.any()):
            return x

        x = x.clone()
        n_time = x.size(-1)
        max_mask_len = max(int(n_time * self.time_mask_max_ratio), 1)
        for batch_idx in torch.nonzero(batch_mask, as_tuple=False).flatten().tolist():
            mask_len = int(torch.randint(1, max_mask_len + 1, (1,)).item())
            start = int(torch.randint(0, max(n_time - mask_len, 1), (1,)).item())
            # Hann window: smooth fade-out rather than hard zeroing
            hann = torch.hann_window(mask_len, device=x.device, dtype=x.dtype)
            # invert: 1 at edges, 0 at centre
            inv_hann = 1.0 - hann
            x[batch_idx, :, start:start + mask_len] *= inv_hann.unsqueeze(0)
        return x

    def _amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel random amplitude scaling to simulate inter-subject variability."""
        if self.amplitude_scale_prob <= 0.0:
            return x
        batch_mask = torch.rand(x.size(0), device=x.device) < self.amplitude_scale_prob
        if not bool(batch_mask.any()):
            return x

        x = x.clone()
        selected = torch.nonzero(batch_mask, as_tuple=False).flatten()
        n_channels = x.size(1)
        # per-channel scale factor in [scale_min, scale_max]
        scales = (
            torch.rand(selected.numel(), n_channels, 1, device=x.device, dtype=x.dtype)
            * (self.amplitude_scale_max - self.amplitude_scale_min)
            + self.amplitude_scale_min
        )
        x[selected] = x[selected] * scales
        return x

    def _freq_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift the frequency spectrum by a small random amount (±max_hz).

        Implemented by rotating the phase of the FFT coefficients in the
        frequency domain, effectively translating spectral content.
        """
        if self.freq_shift_prob <= 0.0 or self.freq_shift_max_hz <= 0.0:
            return x
        batch_mask = torch.rand(x.size(0), device=x.device) < self.freq_shift_prob
        if not bool(batch_mask.any()):
            return x

        x = x.clone()
        n_time = x.size(-1)
        selected = torch.nonzero(batch_mask, as_tuple=False).flatten()
        freqs = torch.fft.rfftfreq(n_time, d=1.0 / self.fs).to(x.device)  # (n_freq,)
        transformed = torch.fft.rfft(x[selected], dim=-1)  # (sel, C, n_freq)

        for local_idx in range(selected.numel()):
            delta_hz = (
                torch.empty(1, device=x.device)
                .uniform_(-self.freq_shift_max_hz, self.freq_shift_max_hz)
                .item()
            )
            # Phase rotation: shift = exp(j * 2π * delta_f * t)
            # In frequency domain, a shift by delta_f corresponds to
            # multiplying by exp(j * 2π * delta_f * n / fs) per time sample,
            # which in the freq domain maps to a circular convolution.
            # Simpler approximation: roll the magnitude spectrum by the
            # nearest integer number of frequency bins.
            freq_resolution = self.fs / n_time
            shift_bins = int(round(delta_hz / freq_resolution))
            if shift_bins == 0:
                continue
            transformed[local_idx] = torch.roll(
                transformed[local_idx], shifts=shift_bins, dims=-1,
            )
            # Zero out the wrapped-around bins to avoid artefacts
            if shift_bins > 0:
                transformed[local_idx, :, :shift_bins] = 0
            else:
                transformed[local_idx, :, shift_bins:] = 0

        x[selected] = torch.fft.irfft(transformed, n=n_time, dim=-1)
        return x

    def _time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Circular temporal shift (jitter) to simulate onset-time uncertainty."""
        if self.time_shift_prob <= 0.0 or self.time_shift_max_samples <= 0:
            return x
        batch_mask = torch.rand(x.size(0), device=x.device) < self.time_shift_prob
        if not bool(batch_mask.any()):
            return x

        x = x.clone()
        for batch_idx in torch.nonzero(batch_mask, as_tuple=False).flatten().tolist():
            shift = int(torch.randint(
                -self.time_shift_max_samples,
                self.time_shift_max_samples + 1,
                (1,),
            ).item())
            if shift != 0:
                x[batch_idx] = torch.roll(x[batch_idx], shifts=shift, dims=-1)
        return x

    # ---- __call__ ----

    def __call__(
        self,
        x: torch.Tensor,
        bipolar_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._gaussian_noise(x)
        x = self._bandstop(x)
        x = self._smooth_time_mask(x)
        x = self._amplitude_scale(x)
        x = self._freq_shift(x)
        x = self._time_shift(x)
        x = self._channel_dropout(x, bipolar_label=bipolar_label)
        return x


class MinorityClassOversampler:
    """Per-batch minority-class oversampling via S&R (Segmentation & Recombination).

    Adapted from EEGConformer's `interaug` method for multi-label SOZ
    localization.  Identifies samples with fewer positive region labels
    (i.e., the "minority" class from the region perspective) and generates
    new synthetic samples by cutting the EEG window into ``n_segments``
    temporal segments and recombining segments from different
    minority-class samples.

    The augmented samples inherit the region / hemisphere / channel labels
    of the *first* donor sample (anchor) to keep labels consistent.
    """

    def __init__(
        self,
        n_segments: int = 8,
        oversample_ratio: float = 0.5,
        region_negative_threshold: float = 0.5,
        augmentor: Optional[EEGWindowAugmentor] = None,
    ):
        self.n_segments = max(int(n_segments), 2)
        self.oversample_ratio = float(max(oversample_ratio, 0.0))
        self.region_negative_threshold = float(region_negative_threshold)
        self.augmentor = augmentor

    def __call__(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        bipolar_label: torch.Tensor,
        monopolar_label: torch.Tensor,
        region_label: torch.Tensor,
        hemisphere_label: torch.Tensor,
        onset_sec: torch.Tensor,
        start_sec: torch.Tensor,
    ):
        """Return the original batch concatenated with synthetic minority samples."""
        if self.oversample_ratio <= 0.0 or x.size(0) == 0:
            return x, label, bipolar_label, monopolar_label, region_label, hemisphere_label, onset_sec, start_sec

        B, C, T = x.shape
        n_regions = region_label.size(1) if region_label.dim() > 1 else 1

        # Identify minority samples: those whose region positive ratio is
        # *below* the threshold.  These are the samples that truly have
        # fewer positive region labels (and thus contribute more "negatives").
        if region_label.dim() > 1:
            pos_ratio = region_label.float().mean(dim=1)  # (B,)
        else:
            pos_ratio = region_label.float()
        minority_mask = pos_ratio < self.region_negative_threshold
        minority_indices = torch.nonzero(minority_mask, as_tuple=False).flatten()

        if minority_indices.numel() < 2:
            # Need at least 2 minority samples to do recombination
            return x, label, bipolar_label, monopolar_label, region_label, hemisphere_label, onset_sec, start_sec

        n_aug = max(int(B * self.oversample_ratio), 1)
        n_aug = min(n_aug, minority_indices.numel())  # don't exceed available
        seg_len = T // self.n_segments
        if seg_len < 1:
            return x, label, bipolar_label, monopolar_label, region_label, hemisphere_label, onset_sec, start_sec

        # Build augmented samples via S&R
        aug_x = torch.zeros(n_aug, C, T, device=x.device, dtype=x.dtype)
        # For labels, use the anchor (first donor) sample
        anchor_indices = minority_indices[torch.randperm(minority_indices.numel(), device=x.device)[:n_aug]]

        for i in range(n_aug):
            anchor_idx = anchor_indices[i].item()
            for seg_idx in range(self.n_segments):
                # Pick a random donor from the minority pool
                donor_idx = minority_indices[
                    torch.randint(0, minority_indices.numel(), (1,), device=x.device).item()
                ].item()
                t_start = seg_idx * seg_len
                t_end = min(t_start + seg_len, T)
                aug_x[i, :, t_start:t_end] = x[donor_idx, :, t_start:t_end]
            # Handle remainder if T is not divisible by n_segments
            remainder_start = self.n_segments * seg_len
            if remainder_start < T:
                aug_x[i, :, remainder_start:] = x[anchor_idx, :, remainder_start:]

        # Apply signal-level augmentation to synthetic samples
        if self.augmentor is not None:
            aug_bipolar = bipolar_label[anchor_indices]
            aug_x = self.augmentor(aug_x, bipolar_label=aug_bipolar)

        # Concatenate augmented samples to the original batch
        x = torch.cat([x, aug_x], dim=0)
        label = torch.cat([label, label[anchor_indices]], dim=0)
        bipolar_label = torch.cat([bipolar_label, bipolar_label[anchor_indices]], dim=0)
        monopolar_label = torch.cat([monopolar_label, monopolar_label[anchor_indices]], dim=0)
        region_label = torch.cat([region_label, region_label[anchor_indices]], dim=0)
        hemisphere_label = torch.cat([hemisphere_label, hemisphere_label[anchor_indices]], dim=0)
        onset_sec = torch.cat([onset_sec, onset_sec[anchor_indices]], dim=0)
        start_sec = torch.cat([start_sec, start_sec[anchor_indices]], dim=0)

        return x, label, bipolar_label, monopolar_label, region_label, hemisphere_label, onset_sec, start_sec



def _resolve_holdout_patient_counts(
    n_patients: int,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    if n_patients < 3:
        raise ValueError(
            f"private_target split requires at least 3 private patients, got {n_patients}"
        )
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_split and test_split must be >= 0")

    n_val = int(round(n_patients * val_ratio)) if val_ratio > 0 else 0
    n_test = int(round(n_patients * test_ratio)) if test_ratio > 0 else 0
    if val_ratio > 0:
        n_val = max(1, n_val)
    if test_ratio > 0:
        n_test = max(1, n_test)

    max_holdout = max(n_patients - 1, 0)
    while n_val + n_test > max_holdout:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    n_train = n_patients - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Invalid private_target split: train={n_train}, val={n_val}, test={n_test}"
        )
    return {'train': n_train, 'val': n_val, 'test': n_test}


def _allocate_group_targets(
    total_target: int,
    group_sizes: Dict[str, int],
) -> Dict[str, int]:
    if total_target <= 0 or not group_sizes:
        return {str(k): 0 for k in group_sizes}

    total = max(sum(group_sizes.values()), 1)
    allocated = {
        str(k): min(int(v), int(np.floor(v * total_target / total)))
        for k, v in group_sizes.items()
    }
    remaining = total_target - sum(allocated.values())
    if remaining <= 0:
        return allocated

    fractions = sorted(
        group_sizes.items(),
        key=lambda kv: (
            (kv[1] * total_target / total) - allocated[str(kv[0])],
            kv[1],
            str(kv[0]),
        ),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key, capacity in fractions:
            key = str(key)
            if allocated[key] >= int(capacity):
                continue
            allocated[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return allocated


def _build_private_patient_infos(private_manifest_ds: ManifestSOZDataset) -> List[Dict[str, object]]:
    patient_infos: List[Dict[str, object]] = []
    grouped = private_manifest_ds.df.groupby('patient_id', sort=False)
    for patient_id, patient_df in grouped:
        hemi_values = [
            str(v).strip()
            for v in patient_df['hemisphere'].tolist()
            if str(v).strip()
        ]
        hemisphere = Counter(hemi_values).most_common(1)[0][0] if hemi_values else 'U'
        patient_infos.append(
            {
                'patient_id': str(patient_id),
                'n_rows': int(len(patient_df)),
                'hemisphere': hemisphere,
            }
        )
    return patient_infos


def _split_private_patients(
    patient_infos: List[Dict[str, object]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    target_counts = _resolve_holdout_patient_counts(
        len(patient_infos),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    total_rows = sum(int(info['n_rows']) for info in patient_infos)
    avg_rows = total_rows / max(len(patient_infos), 1)
    hemisphere_sizes = Counter(str(info['hemisphere']) for info in patient_infos)
    target_hemi = {
        split: _allocate_group_targets(count, hemisphere_sizes)
        for split, count in target_counts.items()
    }
    target_rows = {
        split: total_rows * count / max(len(patient_infos), 1)
        for split, count in target_counts.items()
    }

    rng = np.random.default_rng(seed)
    ordered_patients: List[Dict[str, object]] = []
    for info in patient_infos:
        item = dict(info)
        item['rand'] = float(rng.random())
        ordered_patients.append(item)
    ordered_patients.sort(
        key=lambda item: (-int(item['n_rows']), float(item['rand']), str(item['patient_id']))
    )

    split_order = ('val', 'test', 'train')
    split_rank = {name: idx for idx, name in enumerate(split_order)}
    split_stats = {
        name: {
            'patient_ids': [],
            'n_rows': 0,
            'hemisphere': Counter(),
        }
        for name in target_counts
    }

    for item in ordered_patients:
        choices = [
            split
            for split in split_order
            if len(split_stats[split]['patient_ids']) < target_counts[split]
        ]
        if not choices:
            raise RuntimeError("No available split bucket while assigning private patients")

        best_key = None
        best_split = None
        for split in choices:
            patient_delta = abs(
                (len(split_stats[split]['patient_ids']) + 1) - target_counts[split]
            )
            row_delta = abs(
                (split_stats[split]['n_rows'] + int(item['n_rows'])) - target_rows[split]
            ) / max(avg_rows, 1.0)
            hemisphere = str(item['hemisphere'])
            hemi_delta = abs(
                (split_stats[split]['hemisphere'][hemisphere] + 1)
                - target_hemi[split].get(hemisphere, 0)
            )
            score = patient_delta * 6.0 + row_delta * 1.5 + hemi_delta * 2.5
            candidate_key = (score, split_rank[split])
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_split = split

        assert best_split is not None
        split_stats[best_split]['patient_ids'].append(str(item['patient_id']))
        split_stats[best_split]['n_rows'] += int(item['n_rows'])
        split_stats[best_split]['hemisphere'][str(item['hemisphere'])] += 1

    return {
        split: sorted(stats['patient_ids'])
        for split, stats in split_stats.items()
    }


def _build_private_loo_patient_split(
    patient_infos: List[Dict[str, object]],
    fold_index: int,
    val_offset: int = 1,
) -> Dict[str, List[str]]:
    n_patients = len(patient_infos)
    if n_patients < 3:
        raise ValueError(
            f"private_loo split requires at least 3 private patients, got {n_patients}"
        )

    ordered = sorted(str(info['patient_id']) for info in patient_infos)
    test_idx = int(fold_index) % n_patients
    test_patient = ordered[test_idx]

    if val_offset <= 0:
        raise ValueError(f"private_loo requires val_offset >= 1, got {val_offset}")

    val_idx = (test_idx + int(val_offset)) % n_patients
    if val_idx == test_idx:
        val_idx = (test_idx + 1) % n_patients
    val_patient = ordered[val_idx]

    train_patients = [
        patient_id
        for patient_id in ordered
        if patient_id not in {test_patient, val_patient}
    ]
    if not train_patients:
        raise ValueError("private_loo split produced an empty private train set")

    return {
        'train': train_patients,
        'val': [val_patient],
        'test': [test_patient],
        'fold_index': test_idx,
        'n_folds': n_patients,
        'val_offset': int(val_offset),
    }


def build_soz_datasets(
    args,
    pipeline_cfg,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, Dict[str, object]]:
    split_strategy = args.split_strategy
    private_available = args.source in ('all', 'private')

    if split_strategy == 'auto':
        split_strategy = 'private_target' if private_available else 'random'

    if split_strategy in ('private_target', 'private_loo') and args.source not in ('all', 'private'):
        raise ValueError(
            f"split_strategy='{split_strategy}' requires --source all/private, got {args.source}"
        )

    hemisphere_label_mode = 'lrb'
    region_label_mode = args.region_label_mode

    dataset_kwargs = dict(
        manifest_path=args.manifest,
        private_data_root=args.private_data_root,
        tusz_data_root=args.tusz_data_root,
        label_mode=args.output_mode,
        region_label_mode=region_label_mode,
        hemisphere_label_mode=hemisphere_label_mode,
        pipeline_cfg=pipeline_cfg,
    )

    if split_strategy in ('private_target', 'private_loo'):
        private_all = ManifestSOZDataset(
            source_filter='private',
            **dataset_kwargs,
        )
        if len(private_all) == 0:
            raise ValueError(
                f"split_strategy='{split_strategy}' requires private samples, but none were found"
            )

        patient_infos = _build_private_patient_infos(private_all)
        if split_strategy == 'private_target':
            patient_split = _split_private_patients(
                patient_infos,
                val_ratio=args.val_split,
                test_ratio=args.test_split,
                seed=args.seed,
            )
        else:
            patient_split = _build_private_loo_patient_split(
                patient_infos,
                fold_index=args.private_loo_fold_index,
                val_offset=args.private_loo_val_offset,
            )

        train_parts: List[torch.utils.data.Dataset] = []
        split_meta: Dict[str, object] = {
            'strategy': split_strategy,
            'region_label_mode': region_label_mode,
            'region_names': list(get_region_names(region_label_mode)),
            'hemisphere_label_mode': hemisphere_label_mode,
            'private_patient_split': patient_split,
            'log_lines': [],
        }
        if split_strategy == 'private_loo':
            split_meta['log_lines'].append(
                "private_loo fold="
                f"{patient_split['fold_index'] + 1}/{patient_split['n_folds']} "
                f"(test={patient_split['test'][0]}, val={patient_split['val'][0]})"
            )

        if args.source == 'all':
            tusz_train_manifest = ManifestSOZDataset(
                source_filter='tusz',
                **dataset_kwargs,
            )
            train_parts.append(
                SOZBrainNetworkDataset(
                    tusz_train_manifest,
                    precomputed_dir=args.precomputed_dir,
                )
            )
            tusz_summary = _summarize_manifest_subset(tusz_train_manifest)
            split_meta['train_tusz_summary'] = tusz_summary
            split_meta['log_lines'].append(_format_subset_summary('train/tusz_all', tusz_summary))

        private_train_manifest = ManifestSOZDataset(
            source_filter='private',
            patient_ids=patient_split['train'],
            **dataset_kwargs,
        )
        private_val_manifest = ManifestSOZDataset(
            source_filter='private',
            patient_ids=patient_split['val'],
            **dataset_kwargs,
        )
        private_test_manifest = ManifestSOZDataset(
            source_filter='private',
            patient_ids=patient_split['test'],
            **dataset_kwargs,
        )

        split_meta['train_private_summary'] = _summarize_manifest_subset(private_train_manifest)
        split_meta['val_summary'] = _summarize_manifest_subset(private_val_manifest)
        split_meta['test_summary'] = _summarize_manifest_subset(private_test_manifest)

        train_parts.append(
            SOZBrainNetworkDataset(
                private_train_manifest,
                precomputed_dir=args.precomputed_dir,
            )
        )
        val_ds = SOZBrainNetworkDataset(
            private_val_manifest,
            precomputed_dir=args.precomputed_dir,
        )
        test_ds = SOZBrainNetworkDataset(
            private_test_manifest,
            precomputed_dir=args.precomputed_dir,
        )

        split_meta['log_lines'].append(
            _format_subset_summary('train/private', split_meta['train_private_summary'])
        )
        split_meta['log_lines'].append(
            f"train/private patients={patient_split['train']}"
        )
        split_meta['log_lines'].append(
            _format_subset_summary('val/private', split_meta['val_summary'])
        )
        split_meta['log_lines'].append(
            f"val/private patients={patient_split['val']}"
        )
        split_meta['log_lines'].append(
            _format_subset_summary('test/private', split_meta['test_summary'])
        )
        split_meta['log_lines'].append(
            f"test/private patients={patient_split['test']}"
        )

        if len(train_parts) == 1:
            train_ds = train_parts[0]
        else:
            train_ds = ConcatDataset(train_parts)
        return train_ds, val_ds, test_ds, split_meta

    manifest_ds = ManifestSOZDataset(
        source_filter=args.source,
        **dataset_kwargs,
    )
    dataset = SOZBrainNetworkDataset(manifest_ds, precomputed_dir=args.precomputed_dir)
    n = len(dataset)
    n_test = int(n * args.test_split)
    n_val = int(n * args.val_split)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    split_meta = {
        'strategy': 'random',
        'region_label_mode': region_label_mode,
        'region_names': list(get_region_names(region_label_mode)),
        'hemisphere_label_mode': hemisphere_label_mode,
        'log_lines': [
            _format_subset_summary('all_sources', _summarize_manifest_subset(manifest_ds)),
            f"random_split train={n_train} val={n_val} test={n_test}",
        ],
    }
    return train_ds, val_ds, test_ds, split_meta


# =====================================================================
# Metrics
# =====================================================================

def compute_localization_ranking_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """Ranking metrics better aligned with multi-channel SOZ localization."""
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    metrics: Dict[str, float] = {'mrr': 0.0, 'valid_localization_samples': 0.0}
    if probs.size == 0 or targets.size == 0 or probs.ndim != 2 or targets.ndim != 2:
        for k in ks:
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0
        return metrics

    recall_sums = {k: 0.0 for k in ks}
    precision_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    valid = 0

    for p, t in zip(probs, targets):
        pos_idx = np.flatnonzero(t > 0.5)
        if len(pos_idx) == 0:
            continue
        valid += 1
        order = np.argsort(p)[::-1]
        pos_set = set(pos_idx.tolist())

        first_positive_rank = None
        for rank, idx in enumerate(order, start=1):
            if idx in pos_set:
                first_positive_rank = rank
                break
        if first_positive_rank is not None:
            mrr_sum += 1.0 / first_positive_rank

        for k in ks:
            topk = order[:min(k, len(order))]
            hits = sum(1 for idx in topk if idx in pos_set)
            recall_sums[k] += hits / max(len(pos_idx), 1)
            precision_sums[k] += hits / max(len(topk), 1)

            dcg = 0.0
            for rank, idx in enumerate(topk, start=1):
                if idx in pos_set:
                    dcg += 1.0 / np.log2(rank + 1)
            ideal_hits = min(len(pos_idx), len(topk))
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_sums[k] += dcg / idcg if idcg > 0 else 0.0

    metrics['valid_localization_samples'] = float(valid)
    if valid == 0:
        for k in ks:
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0
        return metrics

    metrics['mrr'] = mrr_sum / valid
    for k in ks:
        metrics[f'recall_at_{k}'] = recall_sums[k] / valid
        metrics[f'precision_at_{k}'] = precision_sums[k] / valid
        metrics[f'ndcg_at_{k}'] = ndcg_sums[k] / valid
    return metrics


def get_auc_valid_mask(targets: np.ndarray) -> np.ndarray:
    """AUC is only defined when a channel has both positive and negative samples."""
    targets = np.asarray(targets)
    if targets.size == 0 or targets.ndim != 2:
        return np.zeros((0,), dtype=bool)
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    return np.logical_and(pos > 0, neg > 0)


def compute_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    if roc_auc_score is None:
        return 0.0
    valid = get_auc_valid_mask(targets)
    if valid.sum() == 0:
        return 0.0
    auc_values: List[float] = []
    for idx in np.where(valid)[0]:
        try:
            auc_values.append(float(roc_auc_score(targets[:, idx], probs[:, idx])))
        except ValueError:
            continue
    return float(np.mean(auc_values)) if auc_values else 0.0


def build_selection_key(
    metrics: Dict[str, float],
    task_training_mode: str = 'multitask',
) -> Tuple[float, float, float, float, float, float]:
    """Build validation-selection priority according to training mode."""
    mode = str(task_training_mode).strip().lower()
    if mode == 'region_only':
        return (
            float(metrics.get('region_acc', 0.0)),
            float(metrics.get('auc', 0.0)),
            float(metrics.get('ndcg_at_3', 0.0)),
            float(metrics.get('mrr', 0.0)),
            float(metrics.get('recall_at_3', metrics.get('top3', 0.0))),
            float(metrics.get('hemisphere_acc', 0.0)),
        )
    if mode == 'hemisphere_only':
        return (
            float(metrics.get('hemisphere_acc', 0.0)),
            float(metrics.get('region_acc', 0.0)),
            float(metrics.get('auc', 0.0)),
            float(metrics.get('ndcg_at_3', 0.0)),
            float(metrics.get('mrr', 0.0)),
            float(metrics.get('recall_at_3', metrics.get('top3', 0.0))),
        )
    return (
        float(metrics.get('auc', 0.0)),
        float(metrics.get('ndcg_at_3', 0.0)),
        float(metrics.get('mrr', 0.0)),
        float(metrics.get('recall_at_3', metrics.get('top3', 0.0))),
        float(metrics.get('region_acc', 0.0)),
        float(metrics.get('hemisphere_acc', 0.0)),
    )


def format_selection_key_text(
    selection_key: Tuple[float, float, float, float, float, float],
    task_training_mode: str = 'multitask',
) -> str:
    mode = str(task_training_mode).strip().lower()
    if mode == 'region_only':
        return (
            f"region_acc={selection_key[0]:.4f}, auc={selection_key[1]:.4f}, "
            f"ndcg3={selection_key[2]:.4f}, mrr={selection_key[3]:.4f}, "
            f"r3={selection_key[4]:.4f}, hemi_acc={selection_key[5]:.4f}"
        )
    if mode == 'hemisphere_only':
        return (
            f"hemi_acc={selection_key[0]:.4f}, region_acc={selection_key[1]:.4f}, "
            f"auc={selection_key[2]:.4f}, ndcg3={selection_key[3]:.4f}, "
            f"mrr={selection_key[4]:.4f}, r3={selection_key[5]:.4f}"
        )
    return (
        f"auc={selection_key[0]:.4f}, ndcg3={selection_key[1]:.4f}, "
        f"mrr={selection_key[2]:.4f}, r3={selection_key[3]:.4f}, "
        f"region_acc={selection_key[4]:.4f}, hemi_acc={selection_key[5]:.4f}"
    )


def parse_brain_network_features(spec: str) -> Tuple[str, ...]:
    raw = str(spec).strip().lower()
    if not raw or raw == 'all':
        return SUPPORTED_BRAIN_NETWORK_FEATURES

    selected: List[str] = []
    for item in raw.split(','):
        name = item.strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_BRAIN_NETWORK_FEATURES:
            raise ValueError(
                f"Unsupported brain-network feature '{name}'. "
                f"Choose from {SUPPORTED_BRAIN_NETWORK_FEATURES} or use 'all'."
            )
        if name not in selected:
            selected.append(name)

    if not selected:
        raise ValueError("At least one brain-network feature must be selected")
    return tuple(selected)


def load_compatible_model_weights(
    model: nn.Module,
    path: str,
    map_location='cpu',
) -> Dict[str, List[str]]:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint does not contain a valid model state dict: {path}")

    own_state = model.state_dict()
    filtered_state: Dict[str, torch.Tensor] = {}
    unexpected_keys: List[str] = []
    for key, value in state.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key not in own_state:
            unexpected_keys.append(clean_key)
            continue
        if own_state[clean_key].shape != value.shape:
            unexpected_keys.append(
                f"{clean_key} (ckpt={tuple(value.shape)} != model={tuple(own_state[clean_key].shape)})"
            )
            continue
        filtered_state[clean_key] = value

    missing_keys = [key for key in own_state.keys() if key not in filtered_state]
    model.load_state_dict(filtered_state, strict=False)
    return {
        'loaded_keys': sorted(filtered_state.keys()),
        'missing_keys': sorted(missing_keys),
        'unexpected_keys': sorted(unexpected_keys),
    }


def compute_multilabel_accuracy(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    if len(probs) == 0:
        return 0.0
    preds = probs >= threshold
    truth = targets >= 0.5
    return float((preds == truth).mean())


def compute_multiclass_accuracy(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100,
) -> float:
    if len(logits) == 0:
        return 0.0
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(axis=1)
    return float((preds[mask] == targets[mask]).mean())


def compute_patch_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[float, int]:
    if logits.numel() == 0:
        return 0.0, 0
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    valid = int(mask.sum().item())
    if valid == 0:
        return 0.0, 0
    correct = (preds[mask] == targets[mask]).float().mean().item()
    return float(correct), valid


def shuffle_stage_batch_patches(
    x: torch.Tensor,
    stage_labels: torch.Tensor,
    patch_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shuffle patch order within each sample and keep labels aligned."""
    if x.dim() != 3 or stage_labels.dim() != 2:
        return x, stage_labels

    batch_size, n_channels, total_len = x.shape
    n_patches = stage_labels.size(1)
    if n_patches <= 1 or patch_len <= 0 or total_len != n_patches * patch_len:
        return x, stage_labels

    patches = x.view(batch_size, n_channels, n_patches, patch_len)
    perms = torch.stack(
        [torch.randperm(n_patches, device=x.device) for _ in range(batch_size)],
        dim=0,
    )
    patch_index = perms.view(batch_size, 1, n_patches, 1).expand(-1, n_channels, -1, patch_len)
    shuffled_x = patches.gather(2, patch_index).reshape(batch_size, n_channels, total_len)
    shuffled_labels = stage_labels.gather(1, perms)
    return shuffled_x, shuffled_labels


def count_trainable_parameters(model) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def compute_binary_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probs.size == 0 or targets.size == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'specificity': 0.0,
            'balanced_acc': 0.0,
            'auc': 0.0,
            'tp': 0.0,
            'fp': 0.0,
            'tn': 0.0,
            'fn': 0.0,
        }

    preds = (probs >= threshold).astype(np.int64)
    tp = float(np.logical_and(preds == 1, targets == 1).sum())
    fp = float(np.logical_and(preds == 1, targets == 0).sum())
    tn = float(np.logical_and(preds == 0, targets == 0).sum())
    fn = float(np.logical_and(preds == 0, targets == 1).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_acc = 0.5 * (recall + specificity)

    auc = 0.0
    if roc_auc_score is not None and np.unique(targets).size > 1:
        try:
            auc = float(roc_auc_score(targets, probs))
        except ValueError:
            auc = 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'balanced_acc': float(balanced_acc),
        'auc': float(auc),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def compute_binary_metrics_from_counts(
    tp: float,
    fp: float,
    tn: float,
    fn: float,
) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_acc = 0.5 * (recall + specificity)
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'balanced_acc': float(balanced_acc),
    }


def estimate_stage_patch_statistics(
    dataset: EEGStagePretrainDataset,
    ignore_index: int = -100,
) -> Dict[str, object]:
    cfg = dataset.pipeline.cfg
    pos = 0
    neg = 0
    ignored = 0

    for sample in dataset.samples:
        labels, _ = assign_patch_binary_labels(
            seizure_start_sec=sample.seizure_start_sec,
            seizure_end_sec=sample.seizure_end_sec,
            window_start_sec=float(sample.center_sec - cfg.pre_onset_sec),
            file_duration_sec=sample.duration_sec,
            n_patches=cfg.n_patches,
            patch_len=cfg.patch_len,
            fs=cfg.target_fs,
            ignore_index=ignore_index,
        )
        pos += int((labels == SEIZURE_LABEL).sum())
        neg += int((labels == NON_SEIZURE_LABEL).sum())
        ignored += int((labels == ignore_index).sum())

    total = pos + neg
    counts = np.array([neg, pos], dtype=np.float64)
    if total > 0:
        class_weight = counts.sum() / np.clip(counts, a_min=1.0, a_max=None)
        class_weight = class_weight / class_weight.mean()
    else:
        class_weight = np.ones(2, dtype=np.float64)

    return {
        'valid_patches': int(total),
        'positive_patches': int(pos),
        'negative_patches': int(neg),
        'ignored_patches': int(ignored),
        'positive_rate': float(pos / total) if total > 0 else 0.0,
        'class_weight': torch.tensor(class_weight, dtype=torch.float32),
    }


def stage_metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == 'loss':
        return -float(metrics['loss'])
    if metric_name == 'acc':
        return float(metrics['patch_acc'])
    if metric_name == 'auc':
        return float(metrics['auc'])
    if metric_name == 'recall':
        return float(metrics['recall'])
    return float(metrics['f1'])


def stage_metric_display_value(metric_value: float, metric_name: str) -> float:
    if metric_name == 'loss':
        return -float(metric_value)
    return float(metric_value)


def summarize_status_counts(status_counts: Dict[str, int], top_k: int = 6) -> str:
    if not status_counts:
        return 'none'
    counter = Counter({str(k): int(v) for k, v in status_counts.items()})
    return ', '.join(f'{key}:{value}' for key, value in counter.most_common(top_k))


def compute_pos_weight(loader, device='cpu') -> torch.Tensor:
    """Compute pos_weight = n_neg / n_pos per channel from the full dataset."""
    pos_sum = None
    total = 0
    for batch in loader:
        y = batch['label']
        if pos_sum is None:
            pos_sum = torch.zeros(y.shape[1], dtype=torch.float64)
        pos_sum += y.sum(dim=0).double()
        total += y.shape[0]
    neg_sum = total - pos_sum
    pw = (neg_sum / pos_sum.clamp(min=1.0)).float()
    pw = pw.clamp(max=50.0)
    global_pos_rate = pos_sum.sum() / (total * pos_sum.shape[0])
    per_ch_rate = pos_sum / total
    log.info(f"pos_weight per channel: min={pw.min():.1f}, max={pw.max():.1f}, "
             f"mean={pw.mean():.1f}, pos_rate={global_pos_rate:.4f}")
    log.info(f"  per-channel pos_rate: {[f'{r:.2f}' for r in per_ch_rate.tolist()]}")

    if global_pos_rate > 0.40:
        log.warning(
            f"  *** LABEL ANOMALY: global pos_rate={global_pos_rate:.3f} (>{40}%) ***\n"
            f"  This means {global_pos_rate*100:.1f}% of all channel-labels are positive.\n"
            f"  Typical SOZ labeling should have ~10-20% positive rate.\n"
            f"  Likely cause: onset_channels in manifest are the UNION across all\n"
            f"  seizure events per file, inflating labels. Check generate_manifest.py\n"
            f"  and ensure per-event onset channels are used, not file-level union."
        )

    # count samples with extreme positive counts
    n_ch = pos_sum.shape[0]
    all_pos_count = 0
    high_pos_count = 0
    for batch in loader:
        y = batch['label']
        ch_pos = y.sum(dim=1)  # per-sample positive channel count
        all_pos_count += (ch_pos == n_ch).sum().item()
        high_pos_count += (ch_pos > n_ch * 0.5).sum().item()
    if all_pos_count > 0:
        log.warning(f"  {all_pos_count}/{total} samples have ALL {n_ch} channels = 1")
    if high_pos_count > total * 0.3:
        log.warning(f"  {high_pos_count}/{total} samples have >50% channels positive")

    return pw.to(device)


def build_generalized_sample_weight(
    label: torch.Tensor,
    device: torch.device,
    pos_ratio_threshold: float = 0.5,
    positive_weight: float = 0.05,
) -> torch.Tensor:
    """Down-weight samples with a high fraction of positive SOZ channels."""
    pos_ratio = label.sum(dim=1) / max(label.shape[1], 1)
    return torch.where(
        pos_ratio > pos_ratio_threshold,
        torch.tensor(positive_weight, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
    )


# =====================================================================
# Dataset wrapper (adds onset / window metadata for patching)
# =====================================================================

class SOZBrainNetworkDataset(torch.utils.data.Dataset):
    """
    Wraps ManifestSOZDataset and provides seizure metadata
    needed by SeizureAlignedAdaptivePatching.
    """

    def __init__(self, manifest_ds: ManifestSOZDataset, precomputed_dir: str = None):
        self.ds = manifest_ds
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None

    def __len__(self):
        return len(self.ds)

    def _get_cache_path(self, idx: int) -> Optional[Path]:
        if not self.precomputed_dir:
            return None
        row = self.ds.df.iloc[idx]
        edf_rel = Path(str(row.get('edf_path', '')))
        start_sec = float(row.get('window_start_sec', 0.0))
        
        # Keep directory structure: precomputed_dir / dir_of_edf / filename_start_sec.npz
        # Using .with_suffix('') to remove the .edf extension before appending
        rel_path = edf_rel.parent / f"{edf_rel.stem}_w{start_sec:.1f}.npz"
        
        cache_file = self.precomputed_dir / rel_path
        # Ensure the subdirectories exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        return cache_file

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # x = sample['data']                         # [22, 20, 100]
        # label = sample['label']                    # [22] or [19]

        x, label, mask, meta, y_bipolar, y_monopolar, y_region, y_hemisphere = sample

        # flatten to [22, 2000] for patching module
        C, P, L = x.shape
        x_flat = x.reshape(C, P * L)

        # extract onset / start from manifest row
        row = self.ds.df.iloc[idx]
        onset_sec = float(row.get('onset_sec', 5.0))
        start_sec = float(row.get('window_start_sec', 0.0))
        
        ret = {
            'idx': idx,
            'x': x_flat,
            'label': label,
            'bipolar_label': y_bipolar,
            'monopolar_label': y_monopolar,
            'region_label': y_region,
            'hemisphere_label': y_hemisphere,
            'onset_sec': onset_sec,
            'start_sec': start_sec,
            'source': row.get('source', 'unknown'),
            'patient_id': row.get('patient_id', 'unknown'),
            'edf_path': row.get('edf_path', ''),
        }
        
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                data = np.load(str(cache_path))
                ret['brain_nets'] = torch.from_numpy(data['brain_nets'])
                ret['valid_patch_counts'] = torch.tensor(data['valid_patch_counts'])
                ret['rel_time'] = torch.from_numpy(data['rel_time'])
            except Exception as e:
                pass # Fallback to online computation if loading fails
                
        return ret


def collate_fn(batch):
    ret = {
        'idx': [b['idx'] for b in batch],
        'x': torch.stack([b['x'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'bipolar_label': torch.stack([b['bipolar_label'] for b in batch]),
        'monopolar_label': torch.stack([b['monopolar_label'] for b in batch]),
        'region_label': torch.stack([b['region_label'] for b in batch]),
        'hemisphere_label': torch.stack([b['hemisphere_label'] for b in batch]),
        'onset_sec': torch.tensor([b['onset_sec'] for b in batch]),
        'start_sec': torch.tensor([b['start_sec'] for b in batch]),
        'source': [b['source'] for b in batch],
        'patient_id': [b['patient_id'] for b in batch],
        'edf_path': [b['edf_path'] for b in batch],
    }
    
    if all('brain_nets' in b for b in batch):
        ret['brain_nets'] = torch.stack([b['brain_nets'] for b in batch])
        ret['valid_patch_counts'] = torch.stack([b['valid_patch_counts'] for b in batch])
        ret['rel_time'] = torch.stack([b['rel_time'] for b in batch])
        
    return ret


# =====================================================================
# Training loops
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    cfg,
    writer=None,
    generalized_pos_ratio_threshold: float = 0.5,
    generalized_sample_weight: float = 0.05,
    train_augmentor: Optional[EEGWindowAugmentor] = None,
    lr_mirror_prob: float = 0.0,
    minority_oversampler: Optional[MinorityClassOversampler] = None,
):
    model.train()
    base = model.module if hasattr(model, 'module') else model
    total_loss, n_batches = 0.0, 0
    all_probs, all_targets = [], []
    all_region_probs, all_region_targets = [], []
    all_hemi_logits, all_hemi_targets = [], []
    loss_sums: Dict[str, float] = {}

    for step, batch in enumerate(loader):
        x = batch['x'].to(device)
        label = batch['label'].to(device)
        bipolar_label = batch['bipolar_label'].to(device)
        monopolar_label = batch['monopolar_label'].to(device)
        region_label = batch['region_label'].to(device)
        hemisphere_label = batch['hemisphere_label'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        
        brain_nets = batch.get('brain_nets', None)
        vp_counts = batch.get('valid_patch_counts', None)
        rel_time = batch.get('rel_time', None)
        
        if brain_nets is not None:
            brain_nets = brain_nets.to(device)
        if vp_counts is not None:
            vp_counts = vp_counts.to(device)
        if rel_time is not None:
            rel_time = rel_time.to(device)
        if lr_mirror_prob > 0.0 and brain_nets is None:
            x, label, bipolar_label, monopolar_label, region_label, hemisphere_label = (
                apply_lateral_mirror_augmentation(
                    x=x,
                    label=label,
                    bipolar_label=bipolar_label,
                    monopolar_label=monopolar_label,
                    region_label=region_label,
                    hemisphere_label=hemisphere_label,
                    mirror_prob=lr_mirror_prob,
                )
            )
        if train_augmentor is not None and brain_nets is None:
            x = train_augmentor(x, bipolar_label=bipolar_label)
        # Minority-class S&R oversampling (batch size may increase)
        if minority_oversampler is not None and brain_nets is None:
            x, label, bipolar_label, monopolar_label, region_label, hemisphere_label, onset, start = (
                minority_oversampler(
                    x, label, bipolar_label, monopolar_label,
                    region_label, hemisphere_label, onset, start,
                )
            )

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(
                x, onset, start,
                valid_patch_counts=vp_counts,
                brain_networks=brain_nets,
                rel_time=rel_time,
            )

            # build aux targets
            vm = DynamicNetworkEvolutionModel._build_valid_mask(
                outputs['valid_patch_counts'],
                outputs['transition_probs'].size(1),
            )
            aux = DynamicNetworkEvolutionModel.compute_auxiliary_targets(
                outputs['seizure_relative_time'], vm,
            )

            sample_weight = build_generalized_sample_weight(
                label=label,
                device=device,
                pos_ratio_threshold=generalized_pos_ratio_threshold,
                positive_weight=generalized_sample_weight,
            )

            loss, losses = base.compute_loss(
                outputs, label,
                region_targets=region_label,
                hemisphere_targets=hemisphere_label,
                transition_targets=aux['transition_targets'].to(device),
                pattern_targets=aux['pattern_targets'].to(device),
                sample_weight=sample_weight,
            )

        # NaN check
        if torch.isnan(loss):
            log.warning(f"NaN loss at epoch {epoch} step {step}, skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().item())
        all_probs.append(outputs['soz_probs'].detach().cpu().numpy())
        all_targets.append(label.cpu().numpy())
        all_region_probs.append(outputs['region_probs'].detach().cpu().numpy())
        all_region_targets.append(region_label.cpu().numpy())
        all_hemi_logits.append(outputs['hemisphere_logits'].detach().cpu().numpy())
        all_hemi_targets.append(hemisphere_label.cpu().numpy())

        # logits monitoring (every 50 steps)
        if step % 50 == 0:
            with torch.no_grad():
                soz_l = outputs['soz_logits'].detach()
                soz_p = outputs['soz_probs'].detach()
                gate_w = outputs.get('gate_weights')
                region_loss = losses.get('region')
                hemisphere_loss = losses.get('hemisphere')
                log.info(
                    f"  [E{epoch} S{step}] "
                    f"logits(min={soz_l.min():.3f}, max={soz_l.max():.3f}, "
                    f"mean={soz_l.mean():.3f}, std={soz_l.std():.3f}) "
                    f"probs(min={soz_p.min():.3f}, max={soz_p.max():.3f}, "
                    f"mean={soz_p.mean():.3f}) "
                    f"loss={loss.item():.4f}"
                )
                if region_loss is not None or hemisphere_loss is not None:
                    log.info(
                        f"           aux(region={region_loss.detach().item():.4f}, "
                        f"hemisphere={hemisphere_loss.detach().item():.4f})"
                    )
                if gate_w is not None:
                    log.info(
                        f"           gate(min={gate_w.min():.3f}, "
                        f"max={gate_w.max():.3f}, mean={gate_w.mean():.3f})"
                    )

    avg_loss = total_loss / max(n_batches, 1)
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    region_probs = np.concatenate(all_region_probs, axis=0)
    region_targets = np.concatenate(all_region_targets, axis=0)
    hemi_logits = np.concatenate(all_hemi_logits, axis=0)
    hemi_targets = np.concatenate(all_hemi_targets, axis=0)
    rank_metrics = compute_localization_ranking_metrics(probs, targets, ks=(1, 3, 5))
    recall_at_1 = rank_metrics['recall_at_1']
    recall_at_3 = rank_metrics['recall_at_3']
    recall_at_5 = rank_metrics['recall_at_5']
    auc = compute_auc(probs, targets) if roc_auc_score else 0.0
    auc_valid_channels = int(get_auc_valid_mask(targets).sum())
    region_acc = compute_multilabel_accuracy(region_probs, region_targets)
    hemisphere_acc = compute_multiclass_accuracy(hemi_logits, hemi_targets)
    avg_losses = {
        f"loss_{name}": value / max(n_batches, 1)
        for name, value in loss_sums.items()
    }

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/recall_at_1', recall_at_1, epoch)
        writer.add_scalar('train/recall_at_3', recall_at_3, epoch)
        writer.add_scalar('train/recall_at_5', recall_at_5, epoch)
        writer.add_scalar('train/precision_at_3', rank_metrics['precision_at_3'], epoch)
        writer.add_scalar('train/ndcg_at_3', rank_metrics['ndcg_at_3'], epoch)
        writer.add_scalar('train/mrr', rank_metrics['mrr'], epoch)
        writer.add_scalar('train/auc', auc, epoch)
        writer.add_scalar('train/region_acc', region_acc, epoch)
        writer.add_scalar('train/hemisphere_acc', hemisphere_acc, epoch)
        for name, value in avg_losses.items():
            writer.add_scalar(f'train/{name}', value, epoch)
        with torch.no_grad():
            for name, param in base.named_parameters():
                if param.grad is not None and param.requires_grad:
                    writer.add_scalar(f'grad_norm/{name}',
                                      param.grad.norm().item(), epoch)

    return {
        'loss': avg_loss,
        'top1': recall_at_1,
        'top3': recall_at_3,
        'top5': recall_at_5,
        'auc': auc,
        'auc_valid_channels': auc_valid_channels,
        'region_acc': region_acc,
        'hemisphere_acc': hemisphere_acc,
        **rank_metrics,
        **avg_losses,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    generalized_pos_ratio_threshold: float = 0.5,
    generalized_sample_weight: float = 0.05,
    collect_sample_info: bool = False,
):
    model.eval()
    base = model.module if hasattr(model, 'module') else model
    all_probs, all_targets, all_logits = [], [], []
    all_patient_ids: List[str] = []
    all_edf_paths: List[str] = []
    all_region_probs, all_region_targets = [], []
    all_hemi_logits, all_hemi_targets = [], []
    all_gate_weights = []
    all_branch_weights = []
    all_valid_patch_counts = []
    all_rel_time = []
    loss_sums: Dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        x = batch['x'].to(device)
        label = batch['label'].to(device)
        region_label = batch['region_label'].to(device)
        hemisphere_label = batch['hemisphere_label'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)

        if collect_sample_info:
            all_patient_ids.extend(batch.get('patient_id', []))
            all_edf_paths.extend(batch.get('edf_path', []))

        brain_nets = batch.get('brain_nets', None)
        vp_counts = batch.get('valid_patch_counts', None)
        rel_time = batch.get('rel_time', None)
        
        if brain_nets is not None:
            brain_nets = brain_nets.to(device)
        if vp_counts is not None:
            vp_counts = vp_counts.to(device)
        if rel_time is not None:
            rel_time = rel_time.to(device)

        out = model(
            x, onset, start,
            valid_patch_counts=vp_counts,
            brain_networks=brain_nets,
            rel_time=rel_time,
        )

        vm = DynamicNetworkEvolutionModel._build_valid_mask(
            out['valid_patch_counts'],
            out['transition_probs'].size(1),
        )
        aux = DynamicNetworkEvolutionModel.compute_auxiliary_targets(
            out['seizure_relative_time'], vm,
        )
        sample_weight = build_generalized_sample_weight(
            label=label,
            device=device,
            pos_ratio_threshold=generalized_pos_ratio_threshold,
            positive_weight=generalized_sample_weight,
        )
        _, losses = base.compute_loss(
            out,
            label,
            region_targets=region_label,
            hemisphere_targets=hemisphere_label,
            transition_targets=aux['transition_targets'].to(device),
            pattern_targets=aux['pattern_targets'].to(device),
            sample_weight=sample_weight,
        )

        all_probs.append(out['soz_probs'].cpu().numpy())
        all_targets.append(label.cpu().numpy())
        all_logits.append(out['soz_logits'].cpu().numpy())
        all_region_probs.append(out['region_probs'].cpu().numpy())
        all_region_targets.append(region_label.cpu().numpy())
        all_hemi_logits.append(out['hemisphere_logits'].cpu().numpy())
        all_hemi_targets.append(hemisphere_label.cpu().numpy())
        gate_w = out.get('gate_weights')
        if gate_w is not None:
            all_gate_weights.append(gate_w.cpu().numpy())
        branch_w = out.get('branch_weights')
        if branch_w is not None:
            all_branch_weights.append(branch_w.cpu().numpy())
        all_valid_patch_counts.append(out['valid_patch_counts'].cpu().numpy())
        all_rel_time.append(out['seizure_relative_time'].cpu().numpy())
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().item())
        n_batches += 1

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    region_probs = np.concatenate(all_region_probs, axis=0)
    region_targets = np.concatenate(all_region_targets, axis=0)
    hemi_logits = np.concatenate(all_hemi_logits, axis=0)
    hemi_targets = np.concatenate(all_hemi_targets, axis=0)
    gate_weights = np.concatenate(all_gate_weights, axis=0) if all_gate_weights else None
    branch_weights = np.concatenate(all_branch_weights, axis=0) if all_branch_weights else None
    valid_patch_counts = np.concatenate(all_valid_patch_counts, axis=0)
    seizure_relative_time = np.concatenate(all_rel_time, axis=0)
    rank_metrics = compute_localization_ranking_metrics(probs, targets, ks=(1, 3, 5))
    recall_at_1 = rank_metrics['recall_at_1']
    recall_at_3 = rank_metrics['recall_at_3']
    recall_at_5 = rank_metrics['recall_at_5']
    auc = compute_auc(probs, targets) if roc_auc_score else 0.0
    auc_valid_channels = int(get_auc_valid_mask(targets).sum())
    region_acc = compute_multilabel_accuracy(region_probs, region_targets)
    hemisphere_acc = compute_multiclass_accuracy(hemi_logits, hemi_targets)
    avg_losses = {
        f"loss_{name}": value / max(n_batches, 1)
        for name, value in loss_sums.items()
    }

    log.info(
        f"  [eval] logits(min={logits.min():.3f}, max={logits.max():.3f}, "
        f"mean={logits.mean():.3f}, std={logits.std():.3f}) "
        f"probs(min={probs.min():.3f}, max={probs.max():.3f}, "
        f"mean={probs.mean():.3f}) "
        f"label_pos_rate={targets.mean():.4f} "
        f"r3={recall_at_3:.3f} "
        f"ndcg3={rank_metrics['ndcg_at_3']:.3f} "
        f"mrr={rank_metrics['mrr']:.3f} "
        f"auc_valid_ch={auc_valid_channels}/{targets.shape[1]} "
        f"region_acc={region_acc:.3f} "
        f"hemi_acc={hemisphere_acc:.3f}"
    )

    result = {
        'top1': recall_at_1,
        'top3': recall_at_3,
        'top5': recall_at_5,
        'auc': auc,
        'auc_valid_channels': auc_valid_channels,
        'region_acc': region_acc,
        'hemisphere_acc': hemisphere_acc,
        **rank_metrics,
        'probs': probs,
        'targets': targets,
        'region_probs': region_probs,
        'region_targets': region_targets,
        'hemisphere_logits': hemi_logits,
        'hemisphere_targets': hemi_targets,
        'gate_weights': gate_weights,
        'branch_weights': branch_weights,
        'valid_patch_counts': valid_patch_counts,
        'seizure_relative_time': seizure_relative_time,
        **avg_losses,
    }
    if collect_sample_info:
        result['patient_ids'] = all_patient_ids
        result['edf_paths'] = all_edf_paths
    return result


def train_stage_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    writer=None,
    show_progress: bool = False,
    log_every: int = 20,
    shuffle_patches: bool = False,
):
    model.train()
    base = model.module if hasattr(model, 'module') else model
    total_loss = 0.0
    total_correct = 0.0
    total_valid = 0
    total_pos = 0
    n_batches = 0
    loss_sums: Dict[str, float] = {}
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    seen_windows = 0
    effective_windows = 0
    skipped_windows = 0
    status_counts: Counter = Counter()

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc=f'stage-train {epoch + 1}',
            leave=False,
            dynamic_ncols=True,
        )

    for step, batch in enumerate(iterator, start=1):
        x = batch['x'].to(device)
        stage_labels = batch['stage_labels'].to(device)
        if shuffle_patches:
            x, stage_labels = shuffle_stage_batch_patches(
                x,
                stage_labels,
                patch_len=base.cfg.patch_len,
            )
        load_status = [str(s) for s in batch.get('load_status', [])]
        stage_valid_count = batch.get('stage_valid_count', None)
        if stage_valid_count is not None:
            batch_valid_counts = stage_valid_count.cpu()
            effective_step_windows = int((batch_valid_counts > 0).sum().item())
            skipped_step_windows = int((batch_valid_counts <= 0).sum().item())
        else:
            batch_valid_counts = (stage_labels != base.cfg.stage_ignore_index).sum(dim=1).cpu()
            effective_step_windows = int((batch_valid_counts > 0).sum().item())
            skipped_step_windows = int((batch_valid_counts <= 0).sum().item())
        seen_windows += len(load_status) if load_status else int(stage_labels.size(0))
        effective_windows += effective_step_windows
        skipped_windows += skipped_step_windows
        if load_status:
            status_counts.update(load_status)

        valid_patches = int((stage_labels != base.cfg.stage_ignore_index).sum().item())
        if valid_patches == 0:
            continue

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(x)
            loss, losses = base.compute_stage_loss(outputs, stage_labels)

        if torch.isnan(loss):
            log.warning("NaN loss at stage epoch %d step %d, skipping", epoch + 1, step)
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optimizer.step()

        acc, n_valid = compute_patch_accuracy(
            outputs['stage_logits'].detach(),
            stage_labels,
            ignore_index=base.cfg.stage_ignore_index,
        )
        valid_mask = stage_labels != base.cfg.stage_ignore_index
        valid_probs = torch.softmax(outputs['stage_logits'].detach(), dim=-1)[..., 1][valid_mask]
        valid_targets = stage_labels[valid_mask]
        total_correct += acc * n_valid
        total_valid += n_valid
        total_pos += int((stage_labels == 1).sum().item())
        total_loss += float(loss.detach().item())
        n_batches += 1
        if valid_probs.numel() > 0:
            all_probs.append(valid_probs.cpu().numpy())
            all_targets.append(valid_targets.cpu().numpy())
            valid_preds = (valid_probs >= 0.5).long()
            tp += float(((valid_preds == 1) & (valid_targets == 1)).sum().item())
            fp += float(((valid_preds == 1) & (valid_targets == 0)).sum().item())
            tn += float(((valid_preds == 0) & (valid_targets == 0)).sum().item())
            fn += float(((valid_preds == 0) & (valid_targets == 1)).sum().item())
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().item())

        running_loss = total_loss / max(n_batches, 1)
        running_acc = total_correct / max(total_valid, 1)
        running_pos = total_pos / max(total_valid, 1)
        running_binary = compute_binary_metrics_from_counts(tp, fp, tn, fn)
        running_f1 = running_binary['f1']
        running_recall = running_binary['recall']
        running_skip_rate = skipped_windows / max(seen_windows, 1)
        if show_progress:
            iterator.set_postfix(
                loss=f'{running_loss:.4f}',
                acc=f'{running_acc:.3f}',
                rec=f'{running_recall:.3f}',
                f1=f'{running_f1:.3f}',
                skip=f'{running_skip_rate:.2%}',
                pos=f'{running_pos:.3f}',
            )
        if log_every > 0 and (step == 1 or step % log_every == 0):
            log.info(
                "  [stage train] epoch=%d step=%d/%d loss=%.4f acc=%.3f rec=%.3f f1=%.3f "
                "pos=%.3f valid_patches=%d seen=%d effective=%d skipped=%d status=%s",
                epoch + 1,
                step,
                len(loader),
                running_loss,
                running_acc,
                running_recall,
                running_f1,
                running_pos,
                total_valid,
                seen_windows,
                effective_windows,
                skipped_windows,
                summarize_status_counts(status_counts),
            )

    avg_loss = total_loss / max(n_batches, 1)
    patch_acc = total_correct / max(total_valid, 1)
    pos_rate = total_pos / max(total_valid, 1)
    binary_metrics = compute_binary_metrics(
        np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float64),
        np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.int64),
    )
    avg_losses = {
        f"loss_{name}": value / max(n_batches, 1)
        for name, value in loss_sums.items()
    }

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/patch_acc', patch_acc, epoch)
        writer.add_scalar('train/positive_rate', pos_rate, epoch)
        writer.add_scalar('train/precision', binary_metrics['precision'], epoch)
        writer.add_scalar('train/recall', binary_metrics['recall'], epoch)
        writer.add_scalar('train/f1', binary_metrics['f1'], epoch)
        writer.add_scalar('train/balanced_acc', binary_metrics['balanced_acc'], epoch)
        writer.add_scalar('train/auc', binary_metrics['auc'], epoch)
        writer.add_scalar('train/valid_patches', total_valid, epoch)
        writer.add_scalar('train/seen_windows', seen_windows, epoch)
        writer.add_scalar('train/effective_windows', effective_windows, epoch)
        writer.add_scalar('train/skipped_windows', skipped_windows, epoch)
        writer.add_scalar('train/skip_rate', skipped_windows / max(seen_windows, 1), epoch)
        writer.add_scalar(
            'train/mean_valid_patches_per_effective_window',
            total_valid / max(effective_windows, 1),
            epoch,
        )
        for name, value in avg_losses.items():
            writer.add_scalar(f'train/{name}', value, epoch)

    return {
        'loss': avg_loss,
        'patch_acc': patch_acc,
        'positive_rate': pos_rate,
        'valid_patches': int(total_valid),
        'seen_windows': int(seen_windows),
        'effective_windows': int(effective_windows),
        'skipped_windows': int(skipped_windows),
        'skip_rate': float(skipped_windows / max(seen_windows, 1)),
        'mean_valid_patches_per_effective_window': float(total_valid / max(effective_windows, 1)),
        'load_status_counts': dict(status_counts),
        **binary_metrics,
        **avg_losses,
    }


@torch.no_grad()
def evaluate_stage(
    model,
    loader,
    device,
    show_progress: bool = False,
    log_every: int = 20,
    collect_temporal: bool = False,
):
    model.eval()
    base = model.module if hasattr(model, 'module') else model
    total_loss = 0.0
    total_correct = 0.0
    total_valid = 0
    total_pos = 0
    n_batches = 0
    loss_sums: Dict[str, float] = {}
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    seen_windows = 0
    effective_windows = 0
    skipped_windows = 0
    status_counts: Counter = Counter()
    # DeepSOZ temporal records (per-patch absolute time + probability)
    if collect_temporal:
        temporal_records: Dict[str, list] = {
            'patient_id': [], 'edf_path': [],
            'seizure_start_sec': [], 'seizure_end_sec': [],
            'patch_abs_start_sec': [], 'prob_seizure': [], 'label': [],
        }
        _patch_dur = float(base.cfg.patch_len) / float(base.cfg.fs)

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc='stage-val',
            leave=False,
            dynamic_ncols=True,
        )

    for step, batch in enumerate(iterator, start=1):
        x = batch['x'].to(device)
        stage_labels = batch['stage_labels'].to(device)
        load_status = [str(s) for s in batch.get('load_status', [])]
        stage_valid_count = batch.get('stage_valid_count', None)
        if stage_valid_count is not None:
            batch_valid_counts = stage_valid_count.cpu()
            effective_step_windows = int((batch_valid_counts > 0).sum().item())
            skipped_step_windows = int((batch_valid_counts <= 0).sum().item())
        else:
            batch_valid_counts = (stage_labels != base.cfg.stage_ignore_index).sum(dim=1).cpu()
            effective_step_windows = int((batch_valid_counts > 0).sum().item())
            skipped_step_windows = int((batch_valid_counts <= 0).sum().item())
        seen_windows += len(load_status) if load_status else int(stage_labels.size(0))
        effective_windows += effective_step_windows
        skipped_windows += skipped_step_windows
        if load_status:
            status_counts.update(load_status)

        valid_patches = int((stage_labels != base.cfg.stage_ignore_index).sum().item())
        if valid_patches == 0:
            continue

        outputs = model(x)
        loss, losses = base.compute_stage_loss(outputs, stage_labels)
        acc, n_valid = compute_patch_accuracy(
            outputs['stage_logits'],
            stage_labels,
            ignore_index=base.cfg.stage_ignore_index,
        )
        valid_mask = stage_labels != base.cfg.stage_ignore_index
        valid_probs = torch.softmax(outputs['stage_logits'], dim=-1)[..., 1][valid_mask]
        valid_targets = stage_labels[valid_mask]
        total_correct += acc * n_valid
        total_valid += n_valid
        total_pos += int((stage_labels == 1).sum().item())
        total_loss += float(loss.detach().item())
        n_batches += 1
        if valid_probs.numel() > 0:
            all_probs.append(valid_probs.cpu().numpy())
            all_targets.append(valid_targets.cpu().numpy())
            valid_preds = (valid_probs >= 0.5).long()
            tp += float(((valid_preds == 1) & (valid_targets == 1)).sum().item())
            fp += float(((valid_preds == 1) & (valid_targets == 0)).sum().item())
            tn += float(((valid_preds == 0) & (valid_targets == 0)).sum().item())
            fn += float(((valid_preds == 0) & (valid_targets == 1)).sum().item())

        # Collect per-patch temporal information for DeepSOZ-style evaluation
        if collect_temporal:
            batch_ws = batch.get('window_start_sec')
            batch_ss = batch.get('seizure_start_sec')
            batch_se = batch.get('seizure_end_sec')
            batch_pids = batch.get('patient_id')
            batch_edfs = batch.get('edf_path')
            if batch_ws is not None and batch_pids is not None:
                ws_np = batch_ws.numpy() if hasattr(batch_ws, 'numpy') else np.array(batch_ws)
                ss_np = batch_ss.numpy() if hasattr(batch_ss, 'numpy') else np.array(batch_ss)
                se_np = batch_se.numpy() if hasattr(batch_se, 'numpy') else np.array(batch_se)
                labels_cpu = stage_labels.cpu()
                probs_all = torch.softmax(outputs['stage_logits'], dim=-1)[..., 1].cpu()
                for i in range(labels_cpu.size(0)):
                    vmask = labels_cpu[i] != base.cfg.stage_ignore_index
                    idxs = vmask.nonzero(as_tuple=True)[0]
                    if len(idxs) == 0:
                        continue
                    starts = float(ws_np[i]) + idxs.float().numpy() * _patch_dur
                    n_p = len(idxs)
                    temporal_records['patch_abs_start_sec'].extend(starts.tolist())
                    temporal_records['prob_seizure'].extend(probs_all[i, idxs].tolist())
                    temporal_records['label'].extend(labels_cpu[i, idxs].tolist())
                    temporal_records['patient_id'].extend([batch_pids[i]] * n_p)
                    temporal_records['edf_path'].extend([batch_edfs[i]] * n_p)
                    temporal_records['seizure_start_sec'].extend([float(ss_np[i])] * n_p)
                    temporal_records['seizure_end_sec'].extend([float(se_np[i])] * n_p)

        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().item())

        running_binary = compute_binary_metrics_from_counts(tp, fp, tn, fn)
        running_f1 = running_binary['f1']
        running_recall = running_binary['recall']
        if hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(
                loss=f'{total_loss / max(n_batches, 1):.4f}',
                acc=f'{total_correct / max(total_valid, 1):.3f}',
                rec=f'{running_recall:.3f}',
                f1=f'{running_f1:.3f}',
                skip=f'{skipped_windows / max(seen_windows, 1):.2%}',
                pos=f'{total_pos / max(total_valid, 1):.3f}',
            )
        if log_every > 0 and (step == 1 or step % log_every == 0):
            log.info(
                "  [stage val] step=%d/%d loss=%.4f acc=%.3f rec=%.3f f1=%.3f "
                "pos=%.3f valid_patches=%d seen=%d effective=%d skipped=%d status=%s",
                step,
                len(loader),
                total_loss / max(n_batches, 1),
                total_correct / max(total_valid, 1),
                running_recall,
                running_f1,
                total_pos / max(total_valid, 1),
                total_valid,
                seen_windows,
                effective_windows,
                skipped_windows,
                summarize_status_counts(status_counts),
            )

    avg_loss = total_loss / max(n_batches, 1)
    patch_acc = total_correct / max(total_valid, 1)
    pos_rate = total_pos / max(total_valid, 1)
    binary_metrics = compute_binary_metrics(
        np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float64),
        np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.int64),
    )
    avg_losses = {
        f"loss_{name}": value / max(n_batches, 1)
        for name, value in loss_sums.items()
    }
    result = {
        'loss': avg_loss,
        'patch_acc': patch_acc,
        'positive_rate': pos_rate,
        'valid_patches': int(total_valid),
        'seen_windows': int(seen_windows),
        'effective_windows': int(effective_windows),
        'skipped_windows': int(skipped_windows),
        'skip_rate': float(skipped_windows / max(seen_windows, 1)),
        'mean_valid_patches_per_effective_window': float(total_valid / max(effective_windows, 1)),
        'load_status_counts': dict(status_counts),
        **binary_metrics,
        **avg_losses,
    }
    if collect_temporal:
        result['temporal_records'] = temporal_records
    return result


def run_stage_pretraining(
    args,
    output_dir: Path,
    device,
    rank: int,
    world: int,
    local_rank: int,
    patch_len: int,
):
    log.info("=== Stage pretraining (binary seizure vs non-seizure) ===")
    support = inspect_stage_annotation_support(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        source_filter='tusz',
    )
    log.info(
        "  Stage support: classes=%s valid_events=%s raw=%s",
        support.get('supported_classes'),
        support.get('n_valid_events'),
        support.get('raw_annotation_counts', {}),
    )

    try:
        from data_preprocess.eeg_pipeline import PipelineConfig
    except ImportError:
        from ..data_preprocess.eeg_pipeline import PipelineConfig

    stage_pre_sec = float(args.stage_pre_onset_sec)
    stage_post_sec = float(args.stage_post_onset_sec)
    if stage_pre_sec <= 0.0 or stage_post_sec <= 0.0:
        raise ValueError(
            f"stage_pre_onset_sec and stage_post_onset_sec must be > 0, got "
            f"{stage_pre_sec} and {stage_post_sec}"
        )
    stage_roles = tuple(
        str(role).strip().lower()
        for role in args.stage_sample_roles
        if str(role).strip()
    )
    if not stage_roles:
        stage_roles = ('onset',)
    stage_n_pre_patches = int(np.ceil(stage_pre_sec / args.patch_duration))
    stage_n_post_patches = int(np.ceil(stage_post_sec / args.patch_duration))
    stage_onset_jitter_sec = max(float(args.stage_onset_jitter_sec), 0.0)
    log.info(
        "  Stage sampling: roles=%s pre=%.1fs post=%.1fs pre_patches=%d post_patches=%d "
        "onset_jitter=%.1fs(train) shuffle=%s",
        list(stage_roles),
        stage_pre_sec,
        stage_post_sec,
        stage_n_pre_patches,
        stage_n_post_patches,
        stage_onset_jitter_sec,
        args.stage_shuffle_patches,
    )

    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=stage_pre_sec,
        post_onset_sec=stage_post_sec,
        n_patches=stage_n_pre_patches + stage_n_post_patches,
        patch_len=patch_len,
    )

    train_ds = EEGStagePretrainDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        pipeline_cfg=pipeline_cfg,
        source_filter='tusz',
        split_filter=['train'],
        roles=stage_roles,
        center_jitter_sec=stage_onset_jitter_sec,
    )
    val_splits = ['dev']
    val_ds = EEGStagePretrainDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        pipeline_cfg=pipeline_cfg,
        source_filter='tusz',
        split_filter=val_splits,
        roles=stage_roles,
        center_jitter_sec=0.0,
    )
    if len(val_ds) == 0:
        val_splits = ['eval']
        val_ds = EEGStagePretrainDataset(
            manifest_path=args.manifest,
            tusz_data_root=args.tusz_data_root,
            pipeline_cfg=pipeline_cfg,
            source_filter='tusz',
            split_filter=val_splits,
            roles=stage_roles,
            center_jitter_sec=0.0,
        )

    train_meta = summarize_stage_dataset(train_ds)
    val_meta = summarize_stage_dataset(val_ds)
    log.info("  Stage train windows: %s", train_meta)
    log.info("  Stage val windows: %s", val_meta)
    train_patch_stats = estimate_stage_patch_statistics(
        train_ds,
        ignore_index=-100,
    )
    val_patch_stats = estimate_stage_patch_statistics(
        val_ds,
        ignore_index=-100,
    )
    log.info(
        "  Stage patch stats(train): valid=%d pos=%d neg=%d pos_rate=%.3f class_weight=%s",
        train_patch_stats['valid_patches'],
        train_patch_stats['positive_patches'],
        train_patch_stats['negative_patches'],
        train_patch_stats['positive_rate'],
        [round(float(x), 4) for x in train_patch_stats['class_weight'].tolist()],
    )
    log.info(
        "  Stage patch stats(val): valid=%d pos=%d neg=%d pos_rate=%.3f",
        val_patch_stats['valid_patches'],
        val_patch_stats['positive_patches'],
        val_patch_stats['negative_patches'],
        val_patch_stats['positive_rate'],
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        log.warning("  Stage pretraining skipped because train/val windows are empty.")
        return None

    if world > 1:
        train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world)
    else:
        train_sampler = RandomSampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=stage_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=stage_collate_fn,
        pin_memory=True,
    )
    log.info(
        "  Stage loaders ready: train_batches=%d val_batches=%d batch_size=%d workers=%d",
        len(train_loader),
        len(val_loader),
        args.batch_size,
        args.workers,
    )

    cfg = IntegrationConfig(
        task_mode='stage_pretrain',
        embed_dim=args.embed_dim,
        patch_len=patch_len,
        n_pre_patches=stage_n_pre_patches,
        n_post_patches=stage_n_post_patches,
        fs=args.fs,
        labram_checkpoint=args.labram_ckpt,
        output_mode=args.output_mode,
        w_region=args.w_region,
        w_hemisphere=args.w_hemisphere,
        n_frozen_layers=0,
    )
    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg).to(device)
    base_model = model
    base_model.configure_stage_pretraining(train_backbone=args.stage_train_backbone)
    if args.stage_use_class_weight:
        class_weight = train_patch_stats['class_weight'].to(device)
        base_model.set_stage_class_weight(class_weight)
    trainable_params, total_params = count_trainable_parameters(base_model)
    log.info(
        "  Stage param setup: train_backbone=%s use_class_weight=%s shuffle_patches=%s "
        "trainable params=%d/%d",
        args.stage_train_backbone,
        args.stage_use_class_weight,
        args.stage_shuffle_patches,
        trainable_params,
        total_params,
    )
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        base_model = model.module

    optimizer = torch.optim.AdamW(
        base_model.get_param_groups(args.stage_lr),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.stage_epochs, 1),
    )
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    writer = SummaryWriter(str(output_dir / 'tb_stage_pretrain')) if (_HAS_TB and is_main(rank)) else None

    best_metric = float('-inf')
    best_epoch = -1
    patience_counter = 0
    best_path = output_dir / 'best_pretrain_ckpt.pth'

    for epoch in range(args.stage_epochs):
        if is_main(rank):
            log.info("  [stage] starting epoch %03d/%03d", epoch + 1, args.stage_epochs)
        if world > 1:
            train_sampler.set_epoch(epoch)

        train_metrics = train_stage_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            writer,
            show_progress=is_main(rank),
            log_every=args.stage_log_every,
            shuffle_patches=args.stage_shuffle_patches,
        )
        val_metrics = evaluate_stage(
            model,
            val_loader,
            device,
            show_progress=is_main(rank),
            log_every=args.stage_log_every,
            collect_temporal=True,
        )
        scheduler.step()

        # Compute DeepSOZ-style metrics on validation set
        deepsoz_metrics: Dict[str, float] = {}
        if is_main(rank) and 'temporal_records' in val_metrics:
            _patch_dur = float(pipeline_cfg.patch_len) / float(pipeline_cfg.target_fs)
            try:
                deepsoz_metrics = compute_deepsoz_stage_metrics(
                    records=val_metrics['temporal_records'],
                    patch_duration_sec=_patch_dur,
                    smoother_kernel_size=getattr(args, 'stage_deepsoz_smoother_kernel', 31),
                    threshold=None,
                    max_fpr_per_hour=120.0,
                )
            except Exception as exc:
                log.warning("DeepSOZ metrics failed at epoch %d: %s", epoch + 1, exc)

        current_metric = stage_metric_value(val_metrics, args.stage_selection_metric)
        best_metric_for_log = stage_metric_display_value(best_metric, args.stage_selection_metric)
        improved = current_metric > best_metric + 1e-6
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            best_metric_for_log = stage_metric_display_value(best_metric, args.stage_selection_metric)
        else:
            patience_counter += 1

        if is_main(rank):
            train_coverage = train_metrics['valid_patches'] / max(train_patch_stats['valid_patches'], 1)
            val_coverage = val_metrics['valid_patches'] / max(val_patch_stats['valid_patches'], 1)
            log.info(
                "  [stage] epoch %03d/%03d "
                "train_loss=%.4f train_acc=%.3f train_rec=%.3f train_f1=%.3f train_auc=%.3f "
                "val_loss=%.4f val_acc=%.3f val_rec=%.3f val_f1=%.3f val_auc=%.3f val_pos=%.3f",
                epoch + 1,
                args.stage_epochs,
                train_metrics['loss'],
                train_metrics['patch_acc'],
                train_metrics['recall'],
                train_metrics['f1'],
                train_metrics['auc'],
                val_metrics['loss'],
                val_metrics['patch_acc'],
                val_metrics['recall'],
                val_metrics['f1'],
                val_metrics['auc'],
                val_metrics['positive_rate'],
            )
            log.info(
                "  [stage data] train_valid=%d/%d coverage=%.3f effective=%d/%d skip_rate=%.3f "
                "status=%s",
                train_metrics['valid_patches'],
                train_patch_stats['valid_patches'],
                train_coverage,
                train_metrics['effective_windows'],
                train_metrics['seen_windows'],
                train_metrics['skip_rate'],
                summarize_status_counts(train_metrics['load_status_counts']),
            )
            log.info(
                "  [stage data] val_valid=%d/%d coverage=%.3f effective=%d/%d skip_rate=%.3f "
                "status=%s",
                val_metrics['valid_patches'],
                val_patch_stats['valid_patches'],
                val_coverage,
                val_metrics['effective_windows'],
                val_metrics['seen_windows'],
                val_metrics['skip_rate'],
                summarize_status_counts(val_metrics['load_status_counts']),
            )
            if train_coverage < 0.8 or val_coverage < 0.8:
                log.warning(
                    "  [stage data] low valid-patch coverage detected "
                    "(train=%.3f, val=%.3f); many windows may be failing in __getitem__",
                    train_coverage,
                    val_coverage,
                )
            if deepsoz_metrics:
                log.info(
                    "  [stage deepsoz] seizure_sens=%.3f fpr/hr=%.1f latency=%.1fs "
                    "auroc=%.3f win_sens=%.3f win_spec=%.3f threshold=%.3f "
                    "detected=%d/%d",
                    deepsoz_metrics.get('seizure_sensitivity', 0.0),
                    deepsoz_metrics.get('fpr_per_hour', 0.0),
                    deepsoz_metrics.get('mean_latency_sec', 0.0),
                    deepsoz_metrics.get('window_auroc', 0.0),
                    deepsoz_metrics.get('window_sensitivity', 0.0),
                    deepsoz_metrics.get('window_specificity', 0.0),
                    deepsoz_metrics.get('optimal_threshold', 0.5),
                    deepsoz_metrics.get('n_seizures_detected', 0),
                    deepsoz_metrics.get('n_seizures_total', 0),
                )
            if writer:
                writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                writer.add_scalar('val/patch_acc', val_metrics['patch_acc'], epoch)
                writer.add_scalar('val/positive_rate', val_metrics['positive_rate'], epoch)
                writer.add_scalar('val/precision', val_metrics['precision'], epoch)
                writer.add_scalar('val/recall', val_metrics['recall'], epoch)
                writer.add_scalar('val/f1', val_metrics['f1'], epoch)
                writer.add_scalar('val/balanced_acc', val_metrics['balanced_acc'], epoch)
                writer.add_scalar('val/auc', val_metrics['auc'], epoch)
                writer.add_scalar('val/valid_patches', val_metrics['valid_patches'], epoch)
                writer.add_scalar('val/seen_windows', val_metrics['seen_windows'], epoch)
                writer.add_scalar('val/effective_windows', val_metrics['effective_windows'], epoch)
                writer.add_scalar('val/skipped_windows', val_metrics['skipped_windows'], epoch)
                writer.add_scalar('val/skip_rate', val_metrics['skip_rate'], epoch)
                writer.add_scalar(
                    'val/mean_valid_patches_per_effective_window',
                    val_metrics['mean_valid_patches_per_effective_window'],
                    epoch,
                )
                writer.add_scalar('val/coverage_vs_static', val_coverage, epoch)
                writer.add_scalar('train/coverage_vs_static', train_coverage, epoch)
                for key in ('loss_stage', 'loss_moe', 'loss_total'):
                    if key in val_metrics:
                        writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
                if deepsoz_metrics:
                    for dkey, dval in deepsoz_metrics.items():
                        if isinstance(dval, (int, float)):
                            writer.add_scalar(f'val_deepsoz/{dkey}', dval, epoch)

            if improved:
                base_model.save_checkpoint(
                    str(best_path),
                    extra={
                        'epoch': epoch,
                        'best_stage_metric': best_metric,
                        'best_stage_metric_name': args.stage_selection_metric,
                        'stage_metrics': {
                            k: v for k, v in val_metrics.items() if k != 'temporal_records'
                        },
                        'deepsoz_metrics': deepsoz_metrics,
                    },
                )
                log.info(
                    "  [stage] new best %s=%.4f at epoch %03d -> %s",
                    args.stage_selection_metric,
                    best_metric_for_log,
                    epoch + 1,
                    best_path,
                )
            else:
                log.info(
                    "  [stage] no improvement in %s for %d epoch(s) "
                    "(best=%.4f @ epoch %03d)",
                    args.stage_selection_metric,
                    patience_counter,
                    best_metric_for_log,
                    best_epoch + 1 if best_epoch >= 0 else 0,
                )
        if args.stage_early_stop_patience > 0 and patience_counter >= args.stage_early_stop_patience:
            if is_main(rank):
                log.info(
                    "  [stage] early stopping triggered at epoch %03d "
                    "(patience=%d, best_%s=%.4f @ epoch %03d)",
                    epoch + 1,
                    args.stage_early_stop_patience,
                    args.stage_selection_metric,
                    best_metric_for_log,
                    best_epoch + 1 if best_epoch >= 0 else 0,
                )
            break

    if is_main(rank):
        log.info(
            "  [stage] finished with best_%s=%.4f at epoch %03d",
            args.stage_selection_metric,
            stage_metric_display_value(best_metric, args.stage_selection_metric),
            best_epoch + 1 if best_epoch >= 0 else 0,
        )
    if writer:
        writer.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_path if best_path.exists() else None


# =====================================================================
# Contrastive pretraining
# =====================================================================

def run_contrastive_pretraining(
    model_pretrain, train_loader, device, args, writer=None,
):
    log.info("=== Contrastive pretraining ===")
    net_ext = MultiScaleBrainNetworkExtractor(
        n_channels=22, patch_len=int(args.patch_duration * args.fs), fs=args.fs,
    ).to(device)
    optimizer = torch.optim.AdamW(model_pretrain.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs,
    )

    for epoch in range(args.pretrain_epochs):
        model_pretrain.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            x = batch['x'].to(device)
            # compute brain networks from raw patches
            B, C, T = x.shape
            P = T // 100
            patches = x.reshape(B, C, P, 100).permute(0, 2, 1, 3)  # [B,P,22,100]
            with torch.no_grad():
                nets = net_ext(patches)['all']                        # [B,P,22,22,4]

            # for contrastive: use same data with augmentation as pos,
            # circularly shifted as neg
            neg = nets.roll(1, dims=0)
            out = model_pretrain(nets, neg)
            loss = out['contrastive_loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_pretrain.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(len(train_loader), 1)
        if writer:
            writer.add_scalar('pretrain/loss', avg, epoch)
        if epoch % 10 == 0:
            log.info(f"  Pretrain epoch {epoch}/{args.pretrain_epochs}  loss={avg:.4f}")

    save_path = Path(args.output_dir) / 'pretrained_encoder.pt'
    model_pretrain.save_pretrained_encoder(str(save_path))
    log.info(f"Pretrained encoder saved to {save_path}")
    return save_path


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description='SOZ Locator with Brain Networks')

    # data
    p.add_argument('--manifest', required=True, help='combined_manifest.csv')
    p.add_argument('--private-data-root', default='', help='preprocessed data root')
    p.add_argument('--tusz-data-root', default='', help='TUSZ EDF root')
    p.add_argument('--source', default='all', choices=['tusz', 'private', 'all'])
    p.add_argument(
        '--split-strategy',
        default='auto',
        choices=['auto', 'random', 'private_target', 'private_loo'],
        help=(
            "Dataset split strategy for SOZ finetuning: "
            "auto=use private patient-wise split when source is all/private, "
            "otherwise random"
        ),
    )
    p.add_argument(
        '--private-loo-fold-index',
        type=int,
        default=0,
        help='When split_strategy=private_loo, hold out this private patient fold as test (0-based, wraps around)',
    )
    p.add_argument(
        '--private-loo-val-offset',
        type=int,
        default=1,
        help='When split_strategy=private_loo, pick the validation patient by offsetting from the test fold',
    )

    # model
    p.add_argument('--labram-ckpt', default='', help='LaBraM pretrained weights')
    p.add_argument('--patch-duration', type=float, default=1.0)
    p.add_argument('--fs', type=float, default=200.0)
    p.add_argument('--embed-dim', type=int, default=200)
    p.add_argument('--labram-frozen-layers', type=int, default=10,
                   help='During SOZ finetuning, keep the bottom N LaBraM transformer blocks frozen; patch/embed and pos/time embeddings stay frozen with them')
    p.add_argument('--output-mode', default='monopolar', choices=['monopolar', 'bipolar'])
    p.add_argument(
        '--region-label-mode',
        default='coarse',
        choices=['coarse', 'fine_lateralized'],
        help='Region target granularity: coarse=[FP,F,C,T,P,O], fine_lateralized=[L_FP,R_FP,L_F,R_F,C,L_T,R_T,P,O]',
    )

    # brain networks
    p.add_argument('--brain-network-features', default='gc,te,aec,wpli')
    p.add_argument('--use-contrastive', action='store_true')
    p.add_argument('--pretrain-epochs', type=int, default=50)
    p.add_argument('--use-pretrain-stage', action='store_true',
                   help='Run binary seizure/non-seizure stage pretraining on TUSZ before SOZ finetuning')
    p.add_argument('--stage-only', action='store_true',
                   help='Run only stage-1 binary seizure/non-seizure pretraining and exit')
    p.add_argument('--stage-pretrain-ckpt', default='',
                   help='Path to a stage-1 checkpoint to load for SOZ-only training')
    p.add_argument('--freeze-labram', action='store_true',
                   help='Freeze LaBraM backbone during SOZ finetuning')
    p.add_argument('--stage-epochs', type=int, default=20,
                   help='Epochs for binary stage pretraining')
    p.add_argument('--stage-lr', type=float, default=1e-4,
                   help='Learning rate for stage pretraining')
    p.add_argument('--stage-log-every', type=int, default=20,
                   help='Log every N steps during stage pretraining')
    p.add_argument('--stage-early-stop-patience', type=int, default=6,
                   help='Early-stop patience for stage pretraining (0 disables)')
    p.add_argument('--stage-selection-metric', default='f1',
                   choices=['f1', 'recall', 'auc', 'acc', 'loss'],
                   help='Validation metric used to save best stage checkpoint and early stop')
    p.add_argument('--stage-train-backbone', dest='stage_train_backbone',
                   action='store_true',
                   help='Train the full LaBraM backbone during stage pretraining')
    p.add_argument('--no-stage-train-backbone', dest='stage_train_backbone',
                   action='store_false',
                   help='Freeze LaBraM backbone and train only the patch classification head')
    p.set_defaults(stage_train_backbone=True)
    p.add_argument('--stage-use-class-weight', dest='stage_use_class_weight',
                   action='store_true',
                   help='Use inverse-frequency class weights for stage CrossEntropy')
    p.add_argument('--no-stage-use-class-weight', dest='stage_use_class_weight',
                   action='store_false',
                   help='Disable class weighting for stage CrossEntropy')
    p.set_defaults(stage_use_class_weight=True)
    p.add_argument('--stage-pre-onset-sec', type=float, default=8.0,
                   help='Seconds before seizure onset used only for stage-1 sampling')
    p.add_argument('--stage-post-onset-sec', type=float, default=4.0,
                   help='Seconds after seizure onset used only for stage-1 sampling')
    p.add_argument('--stage-sample-roles', nargs='+',
                   default=['onset'],
                   choices=['onset', 'mid', 'offset'],
                   help='Stage-1 sampling centers to include')
    p.add_argument('--stage-onset-jitter-sec', type=float, default=3.0,
                   help='Random jitter applied only to stage-1 train onset windows to vary seizure start position; 0 disables')
    p.add_argument('--stage-shuffle-patches', dest='stage_shuffle_patches',
                   action='store_true',
                   help='Randomly shuffle patch order within each stage-1 training sample and shuffle labels in the same way')
    p.add_argument('--no-stage-shuffle-patches', dest='stage_shuffle_patches',
                   action='store_false',
                   help='Keep the original patch order during stage-1 training')
    p.set_defaults(stage_shuffle_patches=True)
    p.add_argument('--stage-deepsoz-smoother-kernel', type=int, default=31,
                   help='Kernel size for DeepSOZ moving average smoother (odd integer, default 31)')
    p.add_argument('--mc-samples', type=int, default=20,
                   help='Number of MC dropout forward passes for DeepSOZ SOZ evaluation (default 20)')
    p.add_argument('--neighbour-threshold', type=int, default=4,
                   help='Max SOZ channels for neighborhood relaxation in DeepSOZ localization (default 4)')

    # Sequence length configurations
    p.add_argument('--pre-onset-sec', type=float, default=5.0, help='Seconds before onset to extract')
    p.add_argument('--post-onset-sec', type=float, default=5.0, help='Seconds after onset to extract')

    # training
    p.add_argument('--finetune-epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--w-transition', type=float, default=0.3,
                   help='Loss weight for patch-level transition detection auxiliary task')
    p.add_argument('--w-pattern', type=float, default=0.2,
                   help='Loss weight for seizure pattern classification auxiliary task')
    p.add_argument('--w-region', type=float, default=0.5,
                   help='Loss weight for coarse region classifier')
    p.add_argument('--w-hemisphere', type=float, default=0.5,
                   help='Loss weight for hemisphere classifier')
    p.add_argument(
        '--task-training-mode',
        choices=('multitask', 'soz_only', 'region_only', 'hemisphere_only'),
        default='multitask',
        help=(
            "Training objective mode: multitask keeps current stage-2 behavior; "
            "soz_only optimizes only the SOZ loss; "
            "region_only optimizes only the region loss; "
            "hemisphere_only optimizes only the hemisphere loss."
        ),
    )
    p.add_argument('--focal-alpha', type=float, default=0.75,
                   help='Positive-class balancing factor for the SOZ focal loss')
    p.add_argument('--focal-gamma', type=float, default=2.0,
                   help='Focusing parameter for the SOZ focal loss')
    p.add_argument('--w-map-pos', type=float, default=0.3,
                   help='Loss weight for MapLossL2PosSum (DeepSOZ-style)')
    p.add_argument('--w-map-neg', type=float, default=0.15,
                   help='Loss weight for MapLossL2Neg (DeepSOZ-style)')
    p.add_argument('--w-map-margin', type=float, default=0.15,
                   help='Loss weight for MapLossMargin (DeepSOZ-style)')
    p.add_argument('--map-margin', type=float, default=0.5,
                   help='Margin value for MapLossMargin')
    p.add_argument('--generalized-pos-ratio-threshold', type=float, default=0.5,
                   help='Samples with positive-channel ratio above this threshold are down-weighted')
    p.add_argument('--generalized-sample-weight', type=float, default=0.05,
                   help='Sample weight applied to samples above --generalized-pos-ratio-threshold')
    p.add_argument('--private-balanced-sampler', dest='private_balanced_sampler',
                   action='store_true',
                   help='For private-only finetuning, use a weighted sampler that balances patients and mildly boosts rare-channel samples')
    p.add_argument('--no-private-balanced-sampler', dest='private_balanced_sampler',
                   action='store_false',
                   help='Disable the private finetuning weighted sampler')
    p.set_defaults(private_balanced_sampler=True)
    p.add_argument('--private-patient-weight-power', type=float, default=1.0,
                   help='Exponent applied to inverse patient-frequency weights in the private balanced sampler')
    p.add_argument('--private-rare-channel-sampler-strength', type=float, default=0.5,
                   help='Mixing factor between patient balancing and rare-channel boosting in the private balanced sampler')
    p.add_argument('--private-rare-channel-sampler-max-boost', type=float, default=2.5,
                   help='Maximum per-sample boost contributed by rare positive channels in the private balanced sampler')
    p.add_argument('--private-sampler-max-weight', type=float, default=4.0,
                   help='Clamp private sampler weights to [1/max_weight, max_weight] before normalization')
    p.add_argument('--private-channel-loss-weight', dest='private_channel_loss_weight',
                   action='store_true',
                   help='For private-only finetuning, reweight SOZ channel loss to protect zero-positive channels and boost rare positive channels')
    p.add_argument('--no-private-channel-loss-weight', dest='private_channel_loss_weight',
                   action='store_false',
                   help='Disable private finetuning channel loss reweighting')
    p.set_defaults(private_channel_loss_weight=True)
    p.add_argument('--private-common-channel-loss-min-weight', type=float, default=0.5,
                   help='Minimum per-channel SOZ loss weight assigned to common private channels')
    p.add_argument('--private-rare-channel-loss-max-weight', type=float, default=3.0,
                   help='Maximum per-channel SOZ loss weight assigned to rare private channels')
    p.add_argument('--private-zero-positive-channel-weight', type=float, default=0.2,
                   help='SOZ loss weight for channels with zero positives in the private finetuning train set')
    p.add_argument('--eeg-augment', dest='eeg_augment',
                   action='store_true',
                   help='Enable EEG signal augmentation during finetuning (all sources)')
    p.add_argument('--no-eeg-augment', dest='eeg_augment',
                   action='store_false',
                   help='Disable EEG augmentation during finetuning')
    # backward compatibility alias
    p.add_argument('--private-eeg-augment', dest='eeg_augment',
                   action='store_true',
                   help=argparse.SUPPRESS)
    p.add_argument('--no-private-eeg-augment', dest='eeg_augment',
                   action='store_false',
                   help=argparse.SUPPRESS)
    p.set_defaults(eeg_augment=True)
    # existing signal augmentations
    p.add_argument('--augment-gaussian-prob', type=float, default=0.4,
                   help='Probability of adding weak Gaussian noise per sample')
    p.add_argument('--augment-gaussian-std-scale', type=float, default=0.01,
                   help='Gaussian noise std as a fraction of each channel standard deviation')
    p.add_argument('--augment-bandstop-prob', type=float, default=0.25,
                   help='Probability of applying a narrow random band-stop filter per sample')
    p.add_argument('--augment-bandstop-min-freq', type=float, default=45.0,
                   help='Minimum center frequency for random band-stop augmentation')
    p.add_argument('--augment-bandstop-max-freq', type=float, default=65.0,
                   help='Maximum center frequency for random band-stop augmentation')
    p.add_argument('--augment-bandstop-width-hz', type=float, default=2.0,
                   help='Bandwidth of the random band-stop augmentation in Hz')
    p.add_argument('--augment-channel-drop-prob', type=float, default=0.15,
                   help='Probability of dropping one or more non-critical bipolar channels')
    p.add_argument('--augment-max-channel-drops', type=int, default=1,
                   help='Maximum number of bipolar channels to drop for each augmented sample')
    p.add_argument('--augment-lr-mirror-prob', type=float, default=0.10,
                   help='Probability of applying left-right mirror augmentation to unilateral samples')
    # new signal augmentations (Rommel et al.)
    p.add_argument('--augment-time-mask-prob', type=float, default=0.3,
                   help='Probability of applying smooth Hann-window time mask')
    p.add_argument('--augment-time-mask-max-ratio', type=float, default=0.2,
                   help='Maximum fraction of temporal length to mask')
    p.add_argument('--augment-amplitude-scale-prob', type=float, default=0.3,
                   help='Probability of per-channel amplitude scaling')
    p.add_argument('--augment-amplitude-scale-min', type=float, default=0.8,
                   help='Minimum amplitude scale factor')
    p.add_argument('--augment-amplitude-scale-max', type=float, default=1.2,
                   help='Maximum amplitude scale factor')
    p.add_argument('--augment-freq-shift-prob', type=float, default=0.2,
                   help='Probability of random frequency spectrum shift')
    p.add_argument('--augment-freq-shift-max-hz', type=float, default=2.0,
                   help='Maximum frequency shift in Hz')
    p.add_argument('--augment-time-shift-prob', type=float, default=0.3,
                   help='Probability of circular temporal shift (jitter)')
    p.add_argument('--augment-time-shift-max-samples', type=int, default=50,
                   help='Maximum circular temporal shift in samples')
    # minority-class S&R oversampling (EEGConformer-inspired)
    p.add_argument('--augment-minority-oversample', type=float, default=0.5,
                   help='Ratio of augmented minority-class samples to add per batch (0 disables)')
    p.add_argument('--augment-sr-segments', type=int, default=8,
                   help='Number of temporal segments for S&R recombination')
    p.add_argument('--augment-minority-region-threshold', type=float, default=0.5,
                   help='Region positive ratio below which a sample is considered minority-class')
    p.add_argument('--amp', action='store_true', help='mixed precision')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)

    # output
    p.add_argument('--output-dir', default='output/train_bn')
    p.add_argument('--precomputed-dir', default=None, help='Directory with precomputed brain networks')
    p.add_argument(
        '--init-soz-ckpt',
        default='',
        help='Initialize the full stage-2 model from a previous SOZ checkpoint, but start a fresh run (unlike --resume)',
    )
    p.add_argument('--resume', default='', help='checkpoint to resume from')
    p.add_argument('--save-every', type=int, default=10)

    # validation
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--test-split', type=float, default=0.15)

    return p.parse_args()


def main():
    args = parse_args()
    selected_brain_features = parse_brain_network_features(args.brain_network_features)
    if args.init_soz_ckpt and args.resume:
        raise ValueError("--init-soz-ckpt and --resume cannot be used together")
    rank, world, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, rank)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Save config ──
    if is_main(rank):
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        log.info(f"Config saved to {output_dir / 'config.json'}")

    # ── 1. Data loading ──
    log.info("=== Step 1: Loading data ===")
    
    # Calculate patch configurations
    patch_len = int(args.patch_duration * args.fs)
    n_pre_patches = int(np.ceil(args.pre_onset_sec / args.patch_duration))
    n_post_patches = int(np.ceil(args.post_onset_sec / args.patch_duration))
    n_patches = n_pre_patches + n_post_patches

    if args.stage_only:
        stage_pretrain_ckpt = run_stage_pretraining(
            args=args,
            output_dir=output_dir,
            device=device,
            rank=rank,
            world=world,
            local_rank=local_rank,
            patch_len=patch_len,
        )
        if is_main(rank):
            log.info("Stage-1 pretraining finished: %s", stage_pretrain_ckpt)
        if world > 1:
            dist.destroy_process_group()
        log.info("Done.")
        return 0
    
    try:
        from data_preprocess.eeg_pipeline import PipelineConfig
    except ImportError:
        from ..data_preprocess.eeg_pipeline import PipelineConfig
        
    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=args.pre_onset_sec,
        post_onset_sec=args.post_onset_sec,
        n_patches=n_patches,
        patch_len=patch_len
    )
    
    if args.source in ('all', 'private') and not args.private_data_root:
        log.warning(
            "  --private-data-root is empty while private samples are enabled; "
            "private EDF relative paths may fail to resolve."
        )

    train_ds, val_ds, test_ds, split_meta = build_soz_datasets(
        args=args,
        pipeline_cfg=pipeline_cfg,
    )
    log.info("  SOZ split strategy: %s", split_meta['strategy'])
    log.info("  Region label mode: %s", split_meta.get('region_label_mode', 'coarse'))
    log.info("  Hemisphere label mode: %s", split_meta.get('hemisphere_label_mode', 'lrb'))
    for line in split_meta.get('log_lines', []):
        log.info("  %s", line)

    n_train = len(train_ds)
    n_val = len(val_ds)
    n_test = len(test_ds)
    log.info("  Final dataset sizes: train=%d, val=%d, test=%d", n_train, n_val, n_test)
    train_analysis = analyze_training_labels(train_ds)
    train_sources = set()
    if train_analysis is not None:
        train_sources = {
            str(src).strip().lower()
            for src in train_analysis['df']['source'].tolist()
        }
        if train_sources == {'private'}:
            private_patient_counts = train_analysis['df']['patient_id'].value_counts().to_dict()
            log.info(
                "  Private finetune train set: patients=%d label_mode=%s",
                int(train_analysis['df']['patient_id'].nunique()),
                train_analysis['label_mode'],
            )
            log.info("  Private finetune patient counts: %s", private_patient_counts)

    # check sample size
    if n_train < 50:
        log.warning(f"  Small dataset ({n_train} samples) — strong augmentation recommended")

    # loaders
    if world > 1:
        train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world)
        if args.private_balanced_sampler and train_sources == {'private'}:
            log.warning("  Private balanced sampler is disabled under DDP; using DistributedSampler instead")
    else:
        train_sampler = RandomSampler(train_ds)
        if args.private_balanced_sampler:
            weighted_sampler, sampler_summary = build_private_weighted_sampler(
                train_analysis,
                patient_power=args.private_patient_weight_power,
                rare_channel_strength=args.private_rare_channel_sampler_strength,
                rare_channel_max_boost=args.private_rare_channel_sampler_max_boost,
                sample_weight_cap=args.private_sampler_max_weight,
            )
            if weighted_sampler is not None:
                train_sampler = weighted_sampler
                log.info(
                    "  Private balanced sampler enabled: weight_min=%.3f weight_max=%.3f weight_mean=%.3f",
                    sampler_summary['weight_min'],
                    sampler_summary['weight_max'],
                    sampler_summary['weight_mean'],
                )
                log.info("  Private sampler patient counts: %s", sampler_summary['patient_counts'])

    train_augmentor = None
    train_lr_mirror_prob = 0.0
    minority_oversampler = None
    if args.eeg_augment:
        if args.precomputed_dir:
            log.warning(
                "  EEG and LR-mirror augmentation disabled because --precomputed-dir "
                "is set; augmenting x would desync cached brain networks."
            )
        else:
            train_lr_mirror_prob = args.augment_lr_mirror_prob
            train_augmentor = EEGWindowAugmentor(
                fs=args.fs,
                gaussian_prob=args.augment_gaussian_prob,
                gaussian_std_scale=args.augment_gaussian_std_scale,
                bandstop_prob=args.augment_bandstop_prob,
                bandstop_min_freq=args.augment_bandstop_min_freq,
                bandstop_max_freq=args.augment_bandstop_max_freq,
                bandstop_width_hz=args.augment_bandstop_width_hz,
                channel_dropout_prob=args.augment_channel_drop_prob,
                max_channel_drops=args.augment_max_channel_drops,
                # new Rommel et al. methods
                time_mask_prob=args.augment_time_mask_prob,
                time_mask_max_ratio=args.augment_time_mask_max_ratio,
                amplitude_scale_prob=args.augment_amplitude_scale_prob,
                amplitude_scale_min=args.augment_amplitude_scale_min,
                amplitude_scale_max=args.augment_amplitude_scale_max,
                freq_shift_prob=args.augment_freq_shift_prob,
                freq_shift_max_hz=args.augment_freq_shift_max_hz,
                time_shift_prob=args.augment_time_shift_prob,
                time_shift_max_samples=args.augment_time_shift_max_samples,
            )
            # Minority-class S&R oversampler
            if args.augment_minority_oversample > 0.0:
                minority_oversampler = MinorityClassOversampler(
                    n_segments=args.augment_sr_segments,
                    oversample_ratio=args.augment_minority_oversample,
                    region_negative_threshold=args.augment_minority_region_threshold,
                    augmentor=train_augmentor,
                )
            log.info(
                "  EEG augmentation enabled: noise(p=%.2f,std_scale=%.4f) "
                "bandstop(p=%.2f,%.1f-%.1fHz,width=%.1fHz) ch_drop(p=%.2f,max=%d) "
                "lr_mirror(p=%.2f) time_mask(p=%.2f,ratio=%.2f) amp_scale(p=%.2f,[%.2f,%.2f]) "
                "freq_shift(p=%.2f,max=%.1fHz) time_shift(p=%.2f,max=%d) "
                "minority_oversample(ratio=%.2f,seg=%d,thresh=%.2f)",
                args.augment_gaussian_prob,
                args.augment_gaussian_std_scale,
                args.augment_bandstop_prob,
                args.augment_bandstop_min_freq,
                args.augment_bandstop_max_freq,
                args.augment_bandstop_width_hz,
                args.augment_channel_drop_prob,
                args.augment_max_channel_drops,
                args.augment_lr_mirror_prob,
                args.augment_time_mask_prob,
                args.augment_time_mask_max_ratio,
                args.augment_amplitude_scale_prob,
                args.augment_amplitude_scale_min,
                args.augment_amplitude_scale_max,
                args.augment_freq_shift_prob,
                args.augment_freq_shift_max_hz,
                args.augment_time_shift_prob,
                args.augment_time_shift_max_samples,
                args.augment_minority_oversample,
                args.augment_sr_segments,
                args.augment_minority_region_threshold,
            )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )

    # ── 2. ModelInit ──
    stage_pretrain_ckpt = None
    if args.stage_pretrain_ckpt:
        candidate = Path(args.stage_pretrain_ckpt)
        if not candidate.exists():
            raise FileNotFoundError(f"Stage pretrain checkpoint not found: {candidate}")
        stage_pretrain_ckpt = candidate
        log.info("Using provided stage-1 checkpoint for SOZ training: %s", stage_pretrain_ckpt)
    elif args.use_pretrain_stage:
        stage_pretrain_ckpt = run_stage_pretraining(
            args=args,
            output_dir=output_dir,
            device=device,
            rank=rank,
            world=world,
            local_rank=local_rank,
            patch_len=patch_len,
        )

    log.info("=== Step 2: Initializing model ===")
    region_names = tuple(split_meta.get('region_names', get_region_names(args.region_label_mode)))
    n_hemisphere_classes = 2 if split_meta.get('hemisphere_label_mode') == 'lr_ignore_b' else 3
    cfg = IntegrationConfig(
        task_mode='soz',
        embed_dim=args.embed_dim,
        patch_len=patch_len,
        n_pre_patches=n_pre_patches,
        n_post_patches=n_post_patches,
        fs=args.fs,
        labram_checkpoint=args.labram_ckpt,
        n_frozen_layers=args.labram_frozen_layers,
        output_mode=args.output_mode,
        n_regions=len(region_names),
        region_label_mode=args.region_label_mode,
        n_hemisphere_classes=n_hemisphere_classes,
        brain_network_features=selected_brain_features,
        w_transition=args.w_transition,
        w_pattern=args.w_pattern,
        w_region=args.w_region,
        w_hemisphere=args.w_hemisphere,
        w_map_pos=args.w_map_pos,
        w_map_neg=args.w_map_neg,
        w_map_margin=args.w_map_margin,
        map_margin=args.map_margin,
        task_training_mode=args.task_training_mode,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )
    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg).to(device)
    log.info(model.summary())
    log.info(
        "  Loss setup: mode=%s w_transition=%.3f w_pattern=%.3f w_region=%.3f "
        "w_hemisphere=%.3f focal_alpha=%.3f focal_gamma=%.3f "
        "w_map_pos=%.3f w_map_neg=%.3f w_map_margin=%.3f map_margin=%.3f "
        "generalized_pos_ratio_threshold=%.3f generalized_sample_weight=%.3f",
        args.task_training_mode,
        args.w_transition,
        args.w_pattern,
        args.w_region,
        args.w_hemisphere,
        args.focal_alpha,
        args.focal_gamma,
        args.w_map_pos,
        args.w_map_neg,
        args.w_map_margin,
        args.map_margin,
        args.generalized_pos_ratio_threshold,
        args.generalized_sample_weight,
    )
    log.info(
        "  Brain-network feature ablation: %s",
        ','.join(selected_brain_features),
    )
    if stage_pretrain_ckpt is not None:
        load_info = model.load_backbone_weights(str(stage_pretrain_ckpt), map_location=device)
        log.info(
            "  Loaded stage-pretrained LaBraM backbone from %s (loaded=%d, missing=%d, unexpected=%d)",
            stage_pretrain_ckpt,
            len(load_info['loaded_keys']),
            len(load_info['missing_keys']),
            len(load_info['unexpected_keys']),
        )
    if args.init_soz_ckpt:
        init_candidate = Path(args.init_soz_ckpt)
        if not init_candidate.exists():
            raise FileNotFoundError(f"SOZ init checkpoint not found: {init_candidate}")
        load_info = load_compatible_model_weights(model, str(init_candidate), map_location=device)
        log.info(
            "  Loaded full SOZ init checkpoint from %s (loaded=%d, missing=%d, unexpected=%d)",
            init_candidate,
            len(load_info['loaded_keys']),
            len(load_info['missing_keys']),
            len(load_info['unexpected_keys']),
        )

    # ── 2b. Compute class balance and set pos_weight ──
    log.info("  Computing pos_weight from training labels...")
    if train_analysis is not None:
        pw = compute_pos_weight_from_analysis(train_analysis, device=device)
    else:
        pw = compute_pos_weight(train_loader, device=device)
    model.set_pos_weight(pw)
    log.info(f"  pos_weight set (shape={pw.shape})")
    if args.private_channel_loss_weight:
        channel_weight, channel_summary = build_private_channel_weight(
            train_analysis,
            min_weight=args.private_common_channel_loss_min_weight,
            max_weight=args.private_rare_channel_loss_max_weight,
            zero_positive_weight=args.private_zero_positive_channel_weight,
            device=device,
        )
        if channel_weight is not None:
            model.set_channel_weight(channel_weight)
            log.info(
                "  Private channel loss weighting enabled: zero_positive=%s",
                channel_summary['zero_positive_channels'],
            )
            log.info(
                "  Private channel loss weights (lowest-count first): %s",
                channel_summary['ranked_channel_weights'],
            )

    # ── 3. Contrastive pretraining (optional) ──
    pretrain_encoder_path = None
    if args.use_contrastive:
        log.info("=== Step 3: Contrastive pretraining ===")
        pt_cfg = PretrainConfig(embed_dim=cfg.embed_dim)
        pretrain_model = BrainNetworkContrastivePretrainer(pt_cfg).to(device)
        writer_pt = SummaryWriter(str(output_dir / 'tb_contrastive_pretrain')) if (_HAS_TB and is_main(rank)) else None
        pretrain_encoder_path = run_contrastive_pretraining(
            pretrain_model, train_loader, device, args, writer_pt,
        )
        if writer_pt:
            writer_pt.close()

    # ── 4. Fine-tuning ──
    log.info("=== Step 4: Fine-tuning ===")
    writer = SummaryWriter(str(output_dir / 'tb_finetune')) if (_HAS_TB and is_main(rank)) else None
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # DDP
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base_model = model.module if hasattr(model, 'module') else model

    # resume
    start_epoch = 0
    best_top3 = 0.0
    best_selection_key = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_top3 = ckpt.get('best_top3', 0.0)
        raw_best_key = ckpt.get('best_selection_key', None)
        if raw_best_key is not None:
            best_selection_key = tuple(float(x) for x in raw_best_key)
            if len(best_selection_key) < 6:
                best_selection_key = best_selection_key + (-1.0,) * (6 - len(best_selection_key))
        else:
            best_selection_key = (0.0, 0.0, 0.0, float(best_top3), 0.0, 0.0)
        log.info(
            "  Resumed from epoch %d, best_recall@3=%.4f, best_key=%s",
            start_epoch,
            best_top3,
            best_selection_key,
        )

    total_epochs = args.finetune_epochs
    phase1_end = total_epochs // 5       # 20% frozen backbone
    phase2_end = total_epochs * 3 // 5   # next 40% unfreeze timefilter

    has_stage_init = stage_pretrain_ckpt is not None

    if args.freeze_labram:
        base_model.freeze_backbone()
        base_model.unfreeze_timefilter()
        log.info("  Finetune setup: LaBraM backbone frozen, TimeFilter + downstream heads trainable")
    elif has_stage_init:
        log.info(
            "  Finetune setup: stage-pretrained LaBraM loaded; bottom %d transformer blocks "
            "+ patch/embed tokens remain frozen, upper LaBraM blocks stay trainable",
            args.labram_frozen_layers,
        )

    optimizer = torch.optim.AdamW(
        base_model.get_param_groups(args.lr), weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2,
    )

    for epoch in range(start_epoch, total_epochs):
        # phase transitions
        if not has_stage_init and not args.freeze_labram:
            if epoch == 0:
                base_model.freeze_backbone()
                log.info("  Phase 1: backbone frozen")
            elif epoch == phase1_end:
                base_model.unfreeze_timefilter()
                optimizer = torch.optim.AdamW(
                    base_model.get_param_groups(args.lr * 0.5), weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=20, T_mult=2,
                )
                log.info("  Phase 2: TimeFilter + network unfrozen")
            elif epoch == phase2_end:
                base_model.unfreeze_all()
                optimizer = torch.optim.AdamW(
                    base_model.get_param_groups(args.lr * 0.1), weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=2,
                )
                log.info("  Phase 3: full model unfrozen")

        if world > 1:
            train_sampler.set_epoch(epoch)

        # train
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            cfg,
            writer,
            generalized_pos_ratio_threshold=args.generalized_pos_ratio_threshold,
            generalized_sample_weight=args.generalized_sample_weight,
            train_augmentor=train_augmentor,
            lr_mirror_prob=train_lr_mirror_prob,
            minority_oversampler=minority_oversampler,
        )
        scheduler.step()
        dt = time.time() - t0

        # validate
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            generalized_pos_ratio_threshold=args.generalized_pos_ratio_threshold,
            generalized_sample_weight=args.generalized_sample_weight,
            collect_sample_info=True,
        )

        # Lightweight per-epoch DeepSOZ SOZ localization metrics
        soz_deepsoz: Dict[str, float] = {}
        if is_main(rank) and 'patient_ids' in val_metrics:
            try:
                soz_deepsoz = compute_deepsoz_soz_metrics(
                    probs=val_metrics['probs'],
                    targets=val_metrics['targets'],
                    patient_ids=val_metrics['patient_ids'],
                    edf_paths=val_metrics['edf_paths'],
                    neighbour_threshold=getattr(args, 'neighbour_threshold', 4),
                )
            except Exception as exc:
                log.warning("DeepSOZ SOZ metrics failed at epoch %d: %s", epoch + 1, exc)

        if is_main(rank):
            val_summary = {
                key: value for key, value in val_metrics.items()
                if not isinstance(value, np.ndarray)
            }
            log.info(
                f"Epoch {epoch:3d}/{total_epochs} "
                f"loss={train_metrics['loss']:.4f} "
                f"soz={train_metrics.get('loss_soz', 0.0):.4f} "
                f"region={train_metrics.get('loss_region', 0.0):.4f} "
                f"hemi={train_metrics.get('loss_hemisphere', 0.0):.4f} "
                f"train_r3={train_metrics['recall_at_3']:.3f} "
                f"train_ndcg3={train_metrics['ndcg_at_3']:.3f} "
                f"train_mrr={train_metrics['mrr']:.3f} "
                f"train_region_acc={train_metrics['region_acc']:.3f} "
                f"train_hemi_acc={train_metrics['hemisphere_acc']:.3f} "
                f"val_r3={val_metrics['recall_at_3']:.3f} "
                f"val_ndcg3={val_metrics['ndcg_at_3']:.3f} "
                f"val_mrr={val_metrics['mrr']:.3f} "
                f"val_auc={val_metrics['auc']:.3f} "
                f"val_region_acc={val_metrics['region_acc']:.3f} "
                f"val_hemi_acc={val_metrics['hemisphere_acc']:.3f} "
                f"({dt:.1f}s)"
            )
            if writer:
                writer.add_scalar('val/recall_at_1', val_metrics['recall_at_1'], epoch)
                writer.add_scalar('val/recall_at_3', val_metrics['recall_at_3'], epoch)
                writer.add_scalar('val/recall_at_5', val_metrics['recall_at_5'], epoch)
                writer.add_scalar('val/precision_at_3', val_metrics['precision_at_3'], epoch)
                writer.add_scalar('val/ndcg_at_3', val_metrics['ndcg_at_3'], epoch)
                writer.add_scalar('val/mrr', val_metrics['mrr'], epoch)
                writer.add_scalar('val/auc', val_metrics['auc'], epoch)
                writer.add_scalar('val/region_acc', val_metrics['region_acc'], epoch)
                writer.add_scalar('val/hemisphere_acc', val_metrics['hemisphere_acc'], epoch)
                for key in ('loss_total', 'loss_soz', 'loss_region', 'loss_hemisphere'):
                    if key in val_metrics:
                        writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
                if soz_deepsoz:
                    for dkey, dval in soz_deepsoz.items():
                        if isinstance(dval, (int, float)):
                            writer.add_scalar(f'val_deepsoz_soz/{dkey}', dval, epoch)
            if soz_deepsoz:
                log.info(
                    "  [soz deepsoz] corr_sz=%.3f acc_pt_w=%.3f acc_pt_s=%.3f "
                    "acc_pt_l=%.3f seizures=%d patients=%d",
                    soz_deepsoz.get('corr_sz', 0.0),
                    soz_deepsoz.get('acc_pt_weighted', 0.0),
                    soz_deepsoz.get('acc_pt_strict', 0.0),
                    soz_deepsoz.get('acc_pt_lenient', 0.0),
                    soz_deepsoz.get('n_seizures', 0),
                    soz_deepsoz.get('n_patients', 0),
                )

            # save best
            val_selection_key = build_selection_key(val_metrics, args.task_training_mode)
            if val_selection_key > best_selection_key:
                best_selection_key = val_selection_key
                best_top3 = val_metrics['recall_at_3']
                base_model.save_checkpoint(
                    str(output_dir / 'best_model.pt'),
                    extra={
                        'epoch': epoch,
                        'best_top3': best_top3,
                        'best_recall_at_3': best_top3,
                        'best_selection_key': list(best_selection_key),
                        'val_metrics': val_summary,
                        'deepsoz_soz_metrics': soz_deepsoz,
                    },
                )
                log.info(
                    "  ** New best (%s): %s",
                    args.task_training_mode,
                    format_selection_key_text(best_selection_key, args.task_training_mode),
                )

            # periodic save
            if (epoch + 1) % args.save_every == 0:
                base_model.save_checkpoint(
                    str(output_dir / f'ckpt_epoch{epoch:03d}.pt'),
                    extra={
                        'epoch': epoch,
                        'best_top3': best_top3,
                        'best_recall_at_3': best_top3,
                        'best_selection_key': list(best_selection_key),
                    },
                )

    # ── 5. Test evaluation ──
    log.info("=== Step 5: Test evaluation ===")
    # load best
    best_path = output_dir / 'best_model.pt'
    if best_path.exists():
        ckpt_best = torch.load(str(best_path), map_location=device)
        base_model.load_state_dict(ckpt_best['model_state'])

    val_metrics_best = evaluate(
        model,
        val_loader,
        device,
        generalized_pos_ratio_threshold=args.generalized_pos_ratio_threshold,
        generalized_sample_weight=args.generalized_sample_weight,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        generalized_pos_ratio_threshold=args.generalized_pos_ratio_threshold,
        generalized_sample_weight=args.generalized_sample_weight,
        collect_sample_info=True,
    )

    # Full MC dropout DeepSOZ SOZ evaluation on test set
    mc_soz_results: Dict[str, object] = {}
    if is_main(rank):
        _mc_samples = getattr(args, 'mc_samples', 20)
        _neighbour_th = getattr(args, 'neighbour_threshold', 4)
        log.info(
            "Running MC dropout SOZ evaluation (mc_samples=%d, neighbour_threshold=%d) ...",
            _mc_samples, _neighbour_th,
        )
        try:
            mc_soz_results = run_detailed_soz_evaluation(
                model, test_loader, device,
                mc_samples=_mc_samples,
                neighbour_threshold=_neighbour_th,
                output_dir=str(output_dir),
            )
            mc_m = mc_soz_results.get('metrics', {})
            log.info(
                "  [MC soz deepsoz] corr_sz=%.3f szunc=%.4f "
                "acc_pt_w=%.3f acc_pt_s=%.3f acc_pt_l=%.3f ptunc=%.4f "
                "seizures=%d patients=%d",
                mc_m.get('corr_sz', 0.0),
                mc_m.get('szunc_mean', 0.0),
                mc_m.get('acc_pt_weighted', 0.0),
                mc_m.get('acc_pt_strict', 0.0),
                mc_m.get('acc_pt_lenient', 0.0),
                mc_m.get('ptunc_mean', 0.0),
                mc_m.get('n_seizures', 0),
                mc_m.get('n_patients', 0),
            )
        except Exception as exc:
            log.warning("MC dropout SOZ evaluation failed: %s", exc)

    if is_main(rank):
        log.info(
            f"\nTest results:\n"
            f"  Recall@1: {test_metrics['recall_at_1']:.4f}\n"
            f"  Recall@3: {test_metrics['recall_at_3']:.4f}\n"
            f"  Recall@5: {test_metrics['recall_at_5']:.4f}\n"
            f"  Precision@3: {test_metrics['precision_at_3']:.4f}\n"
            f"  nDCG@3: {test_metrics['ndcg_at_3']:.4f}\n"
            f"  MRR:   {test_metrics['mrr']:.4f}\n"
            f"  AUC:   {test_metrics['auc']:.4f}\n"
            f"  Region acc: {test_metrics['region_acc']:.4f}\n"
            f"  Hemisphere acc: {test_metrics['hemisphere_acc']:.4f}"
        )

        # save test report (markdown)
        report = (
            f"# SOZ Localization Report\n\n"
            f"## Configuration\n"
            f"- Manifest: `{args.manifest}`\n"
            f"- LaBraM checkpoint: `{args.labram_ckpt}`\n"
            f"- Contrastive pretraining: {args.use_contrastive}\n"
            f"- Stage pretraining: {args.use_pretrain_stage}\n"
            f"- Stage only mode: {args.stage_only}\n"
            f"- Stage init ckpt: `{args.stage_pretrain_ckpt or stage_pretrain_ckpt or ''}`\n"
            f"- Freeze LaBraM backbone: {bool(args.freeze_labram)}\n"
            f"- LaBraM frozen layers during finetune: {'all' if args.freeze_labram else args.labram_frozen_layers}\n"
            f"- Task training mode: {args.task_training_mode}\n"
            f"- Finetune epochs: {total_epochs}\n"
            f"- Output mode: {args.output_mode}\n\n"
            f"- Region label mode: {args.region_label_mode}\n"
            f"- Region names: {', '.join(region_names)}\n\n"
            f"## Test Metrics\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Recall@1 | {test_metrics['recall_at_1']:.4f} |\n"
            f"| Recall@3 | {test_metrics['recall_at_3']:.4f} |\n"
            f"| Recall@5 | {test_metrics['recall_at_5']:.4f} |\n"
            f"| Precision@3 | {test_metrics['precision_at_3']:.4f} |\n"
            f"| nDCG@3 | {test_metrics['ndcg_at_3']:.4f} |\n"
            f"| MRR | {test_metrics['mrr']:.4f} |\n"
            f"| AUC | {test_metrics['auc']:.4f} |\n"
            f"| Region acc | {test_metrics['region_acc']:.4f} |\n"
            f"| Hemisphere acc | {test_metrics['hemisphere_acc']:.4f} |\n\n"
            f"## Best validation key: {format_selection_key_text(best_selection_key, args.task_training_mode)}\n"
        )
        # Append MC dropout DeepSOZ results if available
        if mc_soz_results and 'metrics' in mc_soz_results:
            mc_m = mc_soz_results['metrics']
            report += (
                f"\n## DeepSOZ SOZ Localization (MC dropout, N={getattr(args, 'mc_samples', 20)})\n\n"
                f"### Seizure-level\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| corr_sz (accuracy) | {mc_m.get('corr_sz', 0.0):.4f} |\n"
                f"| szunc (mean max uncertainty) | {mc_m.get('szunc_mean', 0.0):.4f} |\n"
                f"| n_seizures | {mc_m.get('n_seizures', 0)} |\n\n"
                f"### Patient-level\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| acc_pt (weighted) | {mc_m.get('acc_pt_weighted', 0.0):.4f} |\n"
                f"| acc_pt (strict) | {mc_m.get('acc_pt_strict', 0.0):.4f} |\n"
                f"| acc_pt (lenient) | {mc_m.get('acc_pt_lenient', 0.0):.4f} |\n"
                f"| ptunc (mean max uncertainty) | {mc_m.get('ptunc_mean', 0.0):.4f} |\n"
                f"| n_patients | {mc_m.get('n_patients', 0)} |\n\n"
            )
        (output_dir / 'report.md').write_text(report, encoding='utf-8')
        log.info(f"Report saved to {output_dir / 'report.md'}")

        # save test predictions
        np.savez(
            str(output_dir / 'val_predictions.npz'),
            probs=val_metrics_best['probs'],
            targets=val_metrics_best['targets'],
            region_probs=val_metrics_best['region_probs'],
            region_targets=val_metrics_best['region_targets'],
            region_names=np.asarray(region_names),
            hemisphere_logits=val_metrics_best['hemisphere_logits'],
            hemisphere_targets=val_metrics_best['hemisphere_targets'],
            gate_weights=val_metrics_best['gate_weights'],
            branch_weights=val_metrics_best['branch_weights'],
            valid_patch_counts=val_metrics_best['valid_patch_counts'],
            seizure_relative_time=val_metrics_best['seizure_relative_time'],
        )
        np.savez(
            str(output_dir / 'test_predictions.npz'),
            probs=test_metrics['probs'],
            targets=test_metrics['targets'],
            region_probs=test_metrics['region_probs'],
            region_targets=test_metrics['region_targets'],
            region_names=np.asarray(region_names),
            hemisphere_logits=test_metrics['hemisphere_logits'],
            hemisphere_targets=test_metrics['hemisphere_targets'],
            gate_weights=test_metrics['gate_weights'],
            branch_weights=test_metrics['branch_weights'],
            valid_patch_counts=test_metrics['valid_patch_counts'],
            seizure_relative_time=test_metrics['seizure_relative_time'],
        )
        region_md_path, region_csv_path = save_region_confusion_report(
            region_probs=test_metrics['region_probs'],
            region_targets=test_metrics['region_targets'],
            output_dir=output_dir,
            threshold=0.5,
            region_names=region_names,
        )
        log.info("Region confusion report saved to %s", region_md_path)
        log.info("Region confusion CSV saved to %s", region_csv_path)

    if writer:
        writer.close()
    if world > 1:
        dist.destroy_process_group()

    log.info("Done.")
    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    sys.exit(main())
