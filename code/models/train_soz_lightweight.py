#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_soz_lightweight.py

SOZ localization training script using the lightweight 2-layer Transformer
backbone (integration_model_v2) instead of the pretrained LaBraM.

Differences from train_soz_locator_with_brain_networks.py:
  - No LaBraM checkpoint loading
  - Stage-1 pretraining uses only the lightweight 2-layer Transformer
    backbone and a binary patch-level seizure detection head
  - Stage-2 SOZ training can initialize the Transformer backbone from
    the stage-1 checkpoint
  - Simplified SOZ training: all parameters trainable from the start
    (with optional warmup phase)
  - Everything else (data loading, augmentation, loss, evaluation) is reused

Supports: DDP / AMP / checkpoint resume / TensorBoard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

# ── project path ──
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# ── Import the v2 model ──
try:
    from models.integration_model_v2 import (
        Lightweight_Transformer_BrainNetwork_Integration,
        IntegrationConfig,
    )
except ImportError:
    from .integration_model_v2 import (
        Lightweight_Transformer_BrainNetwork_Integration,
        IntegrationConfig,
    )

# ── Reuse helpers from existing training script ──
try:
    from models.train_soz_locator_with_brain_networks import (
        # setup
        setup_logging, setup_ddp, is_main,
        # data
        SOZBrainNetworkDataset, collate_fn, build_soz_datasets,
        analyze_training_labels,
        compute_pos_weight_from_analysis, compute_pos_weight,
        build_private_channel_weight, build_private_weighted_sampler,
        # augmentation
        EEGWindowAugmentor, MinorityClassOversampler,
        # training / evaluation loops
        train_one_epoch, evaluate,
        train_stage_one_epoch, evaluate_stage,
        # metrics
        build_selection_key, format_selection_key_text,
        estimate_stage_patch_statistics,
        stage_metric_value, stage_metric_display_value,
        summarize_status_counts, count_trainable_parameters,
        parse_brain_network_features,
        load_compatible_model_weights,
        SUPPORTED_BRAIN_NETWORK_FEATURES,
    )
    from models.manifest_dataset import get_region_names
    from models.region_confusion import save_region_confusion_report
    from tasks.stage_detection import (
        EEGStagePretrainDataset,
        inspect_stage_annotation_support,
        stage_collate_fn,
        summarize_stage_dataset,
    )
    from tasks.soz_localization_metrics import (
        compute_deepsoz_soz_metrics,
        run_detailed_soz_evaluation,
    )
except ImportError:
    from .train_soz_locator_with_brain_networks import (
        setup_logging, setup_ddp, is_main,
        SOZBrainNetworkDataset, collate_fn, build_soz_datasets,
        analyze_training_labels,
        compute_pos_weight_from_analysis, compute_pos_weight,
        build_private_channel_weight, build_private_weighted_sampler,
        EEGWindowAugmentor, MinorityClassOversampler,
        train_one_epoch, evaluate,
        train_stage_one_epoch, evaluate_stage,
        build_selection_key, format_selection_key_text,
        estimate_stage_patch_statistics,
        stage_metric_value, stage_metric_display_value,
        summarize_status_counts, count_trainable_parameters,
        parse_brain_network_features,
        load_compatible_model_weights,
        SUPPORTED_BRAIN_NETWORK_FEATURES,
    )
    from .manifest_dataset import get_region_names
    from .region_confusion import save_region_confusion_report
    from ..tasks.stage_detection import (
        EEGStagePretrainDataset,
        inspect_stage_annotation_support,
        stage_collate_fn,
        summarize_stage_dataset,
    )
    from ..tasks.soz_localization_metrics import (
        compute_deepsoz_soz_metrics,
        run_detailed_soz_evaluation,
    )

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

log = logging.getLogger('train_lightweight')


# =====================================================================
# Argument Parser (simplified: no LaBraM args; supports lightweight stage pretraining)
# =====================================================================

def parse_args(args=None):
    p = argparse.ArgumentParser(
        description='SOZ Locator with Lightweight 2-layer Transformer + Brain Networks',
    )

    # data
    p.add_argument('--manifest', required=True, help='combined_manifest.csv')
    p.add_argument('--private-data-root', default='', help='preprocessed data root')
    p.add_argument('--tusz-data-root', default='', help='TUSZ EDF root')
    p.add_argument('--source', default='all', choices=['tusz', 'private', 'all'])
    p.add_argument(
        '--split-strategy', default='auto',
        choices=['auto', 'random', 'private_target', 'private_loo'],
    )
    p.add_argument('--private-loo-fold-index', type=int, default=0)
    p.add_argument('--private-loo-val-offset', type=int, default=1)

    # model
    p.add_argument('--embed-dim', type=int, default=200)
    p.add_argument('--out-chans', type=int, default=8,
                   help='TemporalConv output channels')
    p.add_argument('--n-transformer-layers', type=int, default=2,
                   help='Number of Transformer encoder layers (default 2)')
    p.add_argument('--n-heads', type=int, default=8,
                   help='Number of attention heads in the Transformer')
    p.add_argument('--ff-mult', type=float, default=4.0,
                   help='FFN expansion ratio')
    p.add_argument('--transformer-dropout', type=float, default=0.1)
    p.add_argument('--patch-duration', type=float, default=1.0)
    p.add_argument('--fs', type=float, default=200.0)
    p.add_argument('--output-mode', default='monopolar', choices=['monopolar', 'bipolar'])
    p.add_argument(
        '--region-label-mode', default='coarse',
        choices=['coarse', 'fine_lateralized'],
    )

    # TimeFilter
    p.add_argument('--tf-alpha', type=float, default=0.15,
                   help='TimeFilter k-NN sparsification ratio')
    p.add_argument('--tf-n-heads', type=int, default=4,
                   help='Number of TimeFilter graph heads')
    p.add_argument('--top-p', type=float, default=0.5,
                   help='TimeFilter MoE top-p routing threshold')
    p.add_argument('--n-timefilter-blocks', type=int, default=2,
                   help='Number of TimeFilter GraphBlocks')
    p.add_argument('--temporal-k', type=int, default=3,
                   help='TimeFilter temporal mask radius in patch units')
    p.add_argument('--w-moe', type=float, default=0.01,
                   help='MoE auxiliary loss weight')

    # brain networks
    p.add_argument('--brain-network-features', default='gc,te,aec,wpli')
    p.add_argument('--gc-order', type=int, default=20,
                   help='Granger-causality order for online brain-network extraction')
    p.add_argument('--te-n-bins', type=int, default=8,
                   help='Transfer-entropy bin count for online brain-network extraction')
    p.add_argument('--brain-tf-n-blocks', type=int, default=1,
                   help='DirectedBrainTimeFilter block count')
    p.add_argument('--brain-tf-n-heads', type=int, default=4,
                   help='DirectedBrainTimeFilter head count')
    p.add_argument('--brain-tf-hidden', type=int, default=64,
                   help='DirectedBrainTimeFilter hidden dimension')
    p.add_argument('--gru-hidden', type=int, default=128,
                   help='Dynamic network evolution GRU hidden size')
    p.add_argument('--gru-layers', type=int, default=2,
                   help='Dynamic network evolution GRU layer count')
    p.add_argument('--gcn-hidden', type=int, default=64,
                   help='Dynamic network evolution GCN hidden size')
    p.add_argument('--gat-dropout', type=float, default=0.1,
                   help='Dropout shared by TimeFilter and DirectedBrainTimeFilter')
    p.add_argument('--fusion-dropout', type=float, default=0.1,
                   help='Dropout in gated temporal/network fusion')

    # sequence length
    p.add_argument('--pre-onset-sec', type=float, default=5.0)
    p.add_argument('--post-onset-sec', type=float, default=5.0)

    # training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4,
                   help='Peak learning rate (higher than LaBraM since no pretrained weights)')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=5,
                   help='Linear warmup epochs before cosine annealing')

    # stage-1 binary seizure detection pretraining
    p.add_argument('--use-pretrain-stage', action='store_true',
                   help='Run stage-1 binary seizure/non-seizure pretraining before SOZ training')
    p.add_argument('--stage-only', action='store_true',
                   help='Run only stage-1 binary seizure/non-seizure pretraining and exit')
    p.add_argument('--stage-pretrain-ckpt', default='',
                   help='Path to a stage-1 checkpoint; loads only backbone.* into stage-2')
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
                   help='Train the 2-layer Transformer backbone during stage pretraining')
    p.add_argument('--no-stage-train-backbone', dest='stage_train_backbone',
                   action='store_false',
                   help='Freeze the Transformer backbone and train only the binary stage head')
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
                   help='Random jitter for stage-1 train onset windows; 0 disables')
    p.add_argument('--stage-shuffle-patches', dest='stage_shuffle_patches',
                   action='store_true',
                   help='Randomly shuffle patch order within each stage-1 sample and labels together')
    p.add_argument('--no-stage-shuffle-patches', dest='stage_shuffle_patches',
                   action='store_false',
                   help='Keep original patch order during stage-1 training')
    p.set_defaults(stage_shuffle_patches=True)

    # loss weights
    p.add_argument('--w-transition', type=float, default=0.3)
    p.add_argument('--w-pattern', type=float, default=0.2)
    p.add_argument('--w-region', type=float, default=0.5)
    p.add_argument('--w-hemisphere', type=float, default=0.5)
    p.add_argument(
        '--task-training-mode',
        choices=('multitask', 'soz_only', 'region_only', 'hemisphere_only'),
        default='multitask',
    )
    p.add_argument('--focal-alpha', type=float, default=0.75)
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--w-map-pos', type=float, default=0.3)
    p.add_argument('--w-map-neg', type=float, default=0.15)
    p.add_argument('--w-map-margin', type=float, default=0.15)
    p.add_argument('--map-margin', type=float, default=0.5)
    p.add_argument('--generalized-pos-ratio-threshold', type=float, default=0.5)
    p.add_argument('--generalized-sample-weight', type=float, default=0.05)

    # private sampling / weighting
    p.add_argument('--private-balanced-sampler', dest='private_balanced_sampler',
                   action='store_true')
    p.add_argument('--no-private-balanced-sampler', dest='private_balanced_sampler',
                   action='store_false')
    p.set_defaults(private_balanced_sampler=True)
    p.add_argument('--private-patient-weight-power', type=float, default=1.0)
    p.add_argument('--private-rare-channel-sampler-strength', type=float, default=0.5)
    p.add_argument('--private-rare-channel-sampler-max-boost', type=float, default=2.5)
    p.add_argument('--private-sampler-max-weight', type=float, default=4.0)
    p.add_argument('--private-channel-loss-weight', dest='private_channel_loss_weight',
                   action='store_true')
    p.add_argument('--no-private-channel-loss-weight', dest='private_channel_loss_weight',
                   action='store_false')
    p.set_defaults(private_channel_loss_weight=True)
    p.add_argument('--private-common-channel-loss-min-weight', type=float, default=0.5)
    p.add_argument('--private-rare-channel-loss-max-weight', type=float, default=3.0)
    p.add_argument('--private-zero-positive-channel-weight', type=float, default=0.2)

    # augmentation
    p.add_argument('--eeg-augment', dest='eeg_augment', action='store_true')
    p.add_argument('--no-eeg-augment', dest='eeg_augment', action='store_false')
    p.set_defaults(eeg_augment=True)
    p.add_argument('--augment-gaussian-prob', type=float, default=0.4)
    p.add_argument('--augment-gaussian-std-scale', type=float, default=0.01)
    p.add_argument('--augment-bandstop-prob', type=float, default=0.25)
    p.add_argument('--augment-bandstop-min-freq', type=float, default=45.0)
    p.add_argument('--augment-bandstop-max-freq', type=float, default=65.0)
    p.add_argument('--augment-bandstop-width-hz', type=float, default=2.0)
    p.add_argument('--augment-channel-drop-prob', type=float, default=0.15)
    p.add_argument('--augment-max-channel-drops', type=int, default=1)
    p.add_argument('--augment-lr-mirror-prob', type=float, default=0.10)
    p.add_argument('--augment-time-mask-prob', type=float, default=0.3)
    p.add_argument('--augment-time-mask-max-ratio', type=float, default=0.2)
    p.add_argument('--augment-amplitude-scale-prob', type=float, default=0.3)
    p.add_argument('--augment-amplitude-scale-min', type=float, default=0.8)
    p.add_argument('--augment-amplitude-scale-max', type=float, default=1.2)
    p.add_argument('--augment-freq-shift-prob', type=float, default=0.2)
    p.add_argument('--augment-freq-shift-max-hz', type=float, default=2.0)
    p.add_argument('--augment-time-shift-prob', type=float, default=0.3)
    p.add_argument('--augment-time-shift-max-samples', type=int, default=50)
    p.add_argument('--augment-minority-oversample', type=float, default=0.5)
    p.add_argument('--augment-sr-segments', type=int, default=8)
    p.add_argument('--augment-minority-region-threshold', type=float, default=0.5)

    # misc
    p.add_argument('--amp', action='store_true', help='mixed precision')
    p.add_argument('--use-checkpoint', action='store_true',
                   help='Enable activation checkpointing in the backbone/evolution modules')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)

    # output
    p.add_argument('--output-dir', default='output/train_lightweight')
    p.add_argument('--precomputed-dir', default=None)
    p.add_argument('--init-ckpt', default='',
                   help='Initialize model from a previous v2 checkpoint (fresh run)')
    p.add_argument('--resume', default='', help='checkpoint to resume from')
    p.add_argument('--save-every', type=int, default=10)

    # validation
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--test-split', type=float, default=0.15)

    # DeepSOZ evaluation
    p.add_argument('--mc-samples', type=int, default=20)
    p.add_argument('--neighbour-threshold', type=int, default=4)

    return p.parse_args(args)


def validate_args(args) -> None:
    """Fail fast on incompatible lightweight-model settings."""
    if args.patch_duration <= 0:
        raise ValueError("--patch-duration must be positive")
    if args.fs <= 0:
        raise ValueError("--fs must be positive")
    if args.pre_onset_sec < 0 or args.post_onset_sec < 0:
        raise ValueError("--pre-onset-sec and --post-onset-sec must be non-negative")
    if args.pre_onset_sec + args.post_onset_sec <= 0:
        raise ValueError("At least one of --pre-onset-sec/--post-onset-sec must be positive")
    if args.embed_dim % args.n_heads != 0:
        raise ValueError("--embed-dim must be divisible by --n-heads")
    if args.embed_dim % args.tf_n_heads != 0:
        raise ValueError("--embed-dim must be divisible by --tf-n-heads")
    if not (0.0 <= args.tf_alpha <= 1.0):
        raise ValueError("--tf-alpha must be in [0, 1]")
    if not (0.0 <= args.top_p <= 1.0):
        raise ValueError("--top-p must be in [0, 1]")
    if args.n_timefilter_blocks < 0:
        raise ValueError("--n-timefilter-blocks must be non-negative")
    if args.temporal_k < 0:
        raise ValueError("--temporal-k must be non-negative")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")
    if args.stage_epochs < 0:
        raise ValueError("--stage-epochs must be non-negative")
    if args.stage_lr <= 0:
        raise ValueError("--stage-lr must be positive")
    if args.stage_pre_onset_sec <= 0 or args.stage_post_onset_sec <= 0:
        raise ValueError("--stage-pre-onset-sec and --stage-post-onset-sec must be positive")
    if args.stage_only and args.stage_pretrain_ckpt:
        raise ValueError("--stage-only trains a new stage-1 model; do not combine it with --stage-pretrain-ckpt")
    if args.init_ckpt and (args.use_pretrain_stage or args.stage_pretrain_ckpt):
        raise ValueError("--init-ckpt loads a full model; do not combine it with stage backbone initialization")
    if args.resume and (args.use_pretrain_stage or args.stage_pretrain_ckpt or args.stage_only):
        raise ValueError("--resume restores a full stage-2 run; do not combine it with stage-1 options")


# =====================================================================
# Warmup + Cosine Annealing Scheduler
# =====================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for ``warmup_epochs``, then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return [max(base_lr * cosine, self.min_lr) for base_lr in self.base_lrs]


# =====================================================================
# Stage-1 Binary Seizure Detection Pretraining
# =====================================================================

def run_lightweight_stage_pretraining(
    args,
    output_dir: Path,
    device,
    rank: int,
    world: int,
    local_rank: int,
    patch_len: int,
    selected_brain_features: Tuple[str, ...],
) -> Optional[Path]:
    """Train only the lightweight Transformer backbone + binary patch head."""
    log.info("=== Stage 1: binary seizure detection pretraining ===")
    if args.stage_epochs <= 0:
        log.warning("  Stage pretraining skipped because --stage-epochs <= 0")
        return None
    if args.n_transformer_layers != 2:
        log.warning(
            "  Stage-1 will use --n-transformer-layers=%d; the intended lightweight setup is 2 layers.",
            args.n_transformer_layers,
        )

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

    stage_roles = tuple(
        str(role).strip().lower()
        for role in args.stage_sample_roles
        if str(role).strip()
    ) or ('onset',)
    stage_n_pre_patches = int(np.ceil(args.stage_pre_onset_sec / args.patch_duration))
    stage_n_post_patches = int(np.ceil(args.stage_post_onset_sec / args.patch_duration))
    stage_n_patches = stage_n_pre_patches + stage_n_post_patches
    stage_onset_jitter_sec = max(float(args.stage_onset_jitter_sec), 0.0)
    log.info(
        "  Stage sampling: roles=%s pre=%.1fs post=%.1fs pre_patches=%d post_patches=%d "
        "onset_jitter=%.1fs shuffle=%s",
        list(stage_roles),
        args.stage_pre_onset_sec,
        args.stage_post_onset_sec,
        stage_n_pre_patches,
        stage_n_post_patches,
        stage_onset_jitter_sec,
        args.stage_shuffle_patches,
    )

    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=args.stage_pre_onset_sec,
        post_onset_sec=args.stage_post_onset_sec,
        n_patches=stage_n_patches,
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
    train_patch_stats = estimate_stage_patch_statistics(train_ds, ignore_index=-100)
    val_patch_stats = estimate_stage_patch_statistics(val_ds, ignore_index=-100)
    log.info("  Stage train windows: %s", train_meta)
    log.info("  Stage val windows(%s): %s", val_splits, val_meta)
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

    cfg = IntegrationConfig(
        task_mode='stage_pretrain',
        embed_dim=args.embed_dim,
        out_chans=args.out_chans,
        n_transformer_layers=args.n_transformer_layers,
        n_heads_transformer=args.n_heads,
        ff_mult=args.ff_mult,
        transformer_dropout=args.transformer_dropout,
        tf_alpha=args.tf_alpha,
        tf_n_heads=args.tf_n_heads,
        top_p=args.top_p,
        n_timefilter_blocks=args.n_timefilter_blocks,
        temporal_k=args.temporal_k,
        gat_dropout=args.gat_dropout,
        patch_len=patch_len,
        n_pre_patches=stage_n_pre_patches,
        n_post_patches=stage_n_post_patches,
        fs=args.fs,
        output_mode=args.output_mode,
        region_label_mode=args.region_label_mode,
        brain_network_features=selected_brain_features,
        gc_order=args.gc_order,
        te_n_bins=args.te_n_bins,
        brain_tf_n_blocks=args.brain_tf_n_blocks,
        brain_tf_n_heads=args.brain_tf_n_heads,
        brain_tf_hidden=args.brain_tf_hidden,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        gcn_hidden=args.gcn_hidden,
        fusion_dropout=args.fusion_dropout,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        use_checkpoint=args.use_checkpoint,
    )
    model = Lightweight_Transformer_BrainNetwork_Integration(cfg).to(device)
    base_model = model
    base_model.configure_stage_pretraining(train_backbone=args.stage_train_backbone)
    if args.stage_use_class_weight:
        base_model.set_stage_class_weight(train_patch_stats['class_weight'].to(device))

    trainable_params, total_params = count_trainable_parameters(base_model)
    log.info(
        "  Stage model: train_backbone=%s use_class_weight=%s trainable_params=%d/%d",
        args.stage_train_backbone,
        args.stage_use_class_weight,
        trainable_params,
        total_params,
    )
    if trainable_params <= 0:
        raise RuntimeError("Stage pretraining has no trainable parameters")

    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        base_model = model.module

    param_groups = [
        group for group in base_model.get_param_groups(args.stage_lr)
        if len(group.get('params', [])) > 0
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.stage_epochs, 1),
    )
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    writer = SummaryWriter(str(output_dir / 'tb_stage1')) if (_HAS_TB and is_main(rank)) else None

    best_metric = float('-inf')
    best_epoch = -1
    patience_counter = 0
    best_path = output_dir / 'best_stage1_transformer.pt'

    for epoch in range(args.stage_epochs):
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
            collect_temporal=False,
        )
        scheduler.step()

        current_metric = stage_metric_value(val_metrics, args.stage_selection_metric)
        improved = current_metric > best_metric + 1e-6
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if is_main(rank):
            train_coverage = train_metrics['valid_patches'] / max(train_patch_stats['valid_patches'], 1)
            val_coverage = val_metrics['valid_patches'] / max(val_patch_stats['valid_patches'], 1)
            log.info(
                "  [stage1] epoch %03d/%03d "
                "train_loss=%.4f train_acc=%.3f train_rec=%.3f train_f1=%.3f train_auc=%.3f "
                "val_loss=%.4f val_acc=%.3f val_rec=%.3f val_f1=%.3f val_auc=%.3f",
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
            )
            log.info(
                "  [stage1 data] train_valid=%d/%d coverage=%.3f effective=%d/%d "
                "skip_rate=%.3f status=%s",
                train_metrics['valid_patches'],
                train_patch_stats['valid_patches'],
                train_coverage,
                train_metrics['effective_windows'],
                train_metrics['seen_windows'],
                train_metrics['skip_rate'],
                summarize_status_counts(train_metrics['load_status_counts']),
            )
            log.info(
                "  [stage1 data] val_valid=%d/%d coverage=%.3f effective=%d/%d "
                "skip_rate=%.3f status=%s",
                val_metrics['valid_patches'],
                val_patch_stats['valid_patches'],
                val_coverage,
                val_metrics['effective_windows'],
                val_metrics['seen_windows'],
                val_metrics['skip_rate'],
                summarize_status_counts(val_metrics['load_status_counts']),
            )
            if writer:
                writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                writer.add_scalar('val/patch_acc', val_metrics['patch_acc'], epoch)
                writer.add_scalar('val/precision', val_metrics['precision'], epoch)
                writer.add_scalar('val/recall', val_metrics['recall'], epoch)
                writer.add_scalar('val/f1', val_metrics['f1'], epoch)
                writer.add_scalar('val/balanced_acc', val_metrics['balanced_acc'], epoch)
                writer.add_scalar('val/auc', val_metrics['auc'], epoch)
                writer.add_scalar('val/coverage_vs_static', val_coverage, epoch)
                writer.add_scalar('train/coverage_vs_static', train_coverage, epoch)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

            if improved:
                base_model.save_checkpoint(
                    str(best_path),
                    extra={
                        'epoch': epoch,
                        'stage_metric': best_metric,
                        'stage_metric_name': args.stage_selection_metric,
                        'stage_metrics': val_metrics,
                        'stage_train_patch_stats': train_patch_stats,
                        'stage_val_patch_stats': val_patch_stats,
                    },
                )
                log.info(
                    "  [stage1] new best %s=%.4f at epoch %03d -> %s",
                    args.stage_selection_metric,
                    stage_metric_display_value(best_metric, args.stage_selection_metric),
                    epoch + 1,
                    best_path,
                )
            else:
                log.info(
                    "  [stage1] no improvement in %s for %d epoch(s) "
                    "(best=%.4f @ epoch %03d)",
                    args.stage_selection_metric,
                    patience_counter,
                    stage_metric_display_value(best_metric, args.stage_selection_metric),
                    best_epoch + 1 if best_epoch >= 0 else 0,
                )

        if args.stage_early_stop_patience > 0 and patience_counter >= args.stage_early_stop_patience:
            if is_main(rank):
                log.info(
                    "  [stage1] early stopping at epoch %03d (patience=%d)",
                    epoch + 1,
                    args.stage_early_stop_patience,
                )
            break

    if writer:
        writer.close()
    if world > 1:
        dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_main(rank):
        log.info(
            "  [stage1] finished with best_%s=%.4f at epoch %03d",
            args.stage_selection_metric,
            stage_metric_display_value(best_metric, args.stage_selection_metric),
            best_epoch + 1 if best_epoch >= 0 else 0,
        )
    return best_path if best_path.exists() else None


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    validate_args(args)
    selected_brain_features = parse_brain_network_features(args.brain_network_features)

    if args.init_ckpt and args.resume:
        raise ValueError("--init-ckpt and --resume cannot be used together")

    rank, world, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if is_main(rank):
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        log.info("Config saved to %s", output_dir / 'config.json')

    # ── 1. Data loading ──
    log.info("=== Step 1: Loading data ===")

    patch_len_float = args.patch_duration * args.fs
    patch_len = int(round(patch_len_float))
    if abs(patch_len - patch_len_float) > 1e-6:
        log.warning(
            "  patch_duration * fs is non-integer (%.6f); using rounded patch_len=%d",
            patch_len_float,
            patch_len,
        )
    if patch_len <= 0:
        raise ValueError(f"Invalid patch_len={patch_len}; check --patch-duration and --fs")
    n_pre_patches = int(np.ceil(args.pre_onset_sec / args.patch_duration))
    n_post_patches = int(np.ceil(args.post_onset_sec / args.patch_duration))
    n_patches = n_pre_patches + n_post_patches

    try:
        from data_preprocess.eeg_pipeline import PipelineConfig
    except ImportError:
        from ..data_preprocess.eeg_pipeline import PipelineConfig

    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=args.pre_onset_sec,
        post_onset_sec=args.post_onset_sec,
        n_patches=n_patches,
        patch_len=patch_len,
    )

    stage_pretrain_path: Optional[Path] = None
    if args.stage_pretrain_ckpt:
        stage_pretrain_path = Path(args.stage_pretrain_ckpt)
        if not stage_pretrain_path.exists():
            raise FileNotFoundError(f"Stage pretrain checkpoint not found: {stage_pretrain_path}")
        log.info("  Using existing stage-1 checkpoint: %s", stage_pretrain_path)
    elif args.use_pretrain_stage or args.stage_only:
        stage_pretrain_path = run_lightweight_stage_pretraining(
            args=args,
            output_dir=output_dir,
            device=device,
            rank=rank,
            world=world,
            local_rank=local_rank,
            patch_len=patch_len,
            selected_brain_features=selected_brain_features,
        )
        if stage_pretrain_path is None:
            log.warning("  Stage-1 pretraining did not produce a checkpoint.")

    if args.stage_only:
        if world > 1:
            dist.destroy_process_group()
        log.info("Done stage-only run.")
        return 0

    train_ds, val_ds, test_ds, split_meta = build_soz_datasets(
        args=args,
        pipeline_cfg=pipeline_cfg,
    )
    log.info("  Split strategy: %s", split_meta['strategy'])
    for line in split_meta.get('log_lines', []):
        log.info("  %s", line)

    n_train = len(train_ds)
    n_val = len(val_ds)
    n_test = len(test_ds)
    log.info("  Dataset sizes: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    train_analysis = analyze_training_labels(train_ds)
    train_sources = set()
    if train_analysis is not None:
        train_sources = {
            str(src).strip().lower()
            for src in train_analysis['df']['source'].tolist()
        }

    # Sampler
    if world > 1:
        train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world)
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
                log.info("  Private balanced sampler enabled")

    # Augmentation
    train_augmentor = None
    train_lr_mirror_prob = 0.0
    minority_oversampler = None
    if args.eeg_augment and not args.precomputed_dir:
        train_lr_mirror_prob = args.augment_lr_mirror_prob
        train_augmentor = EEGWindowAugmentor(
            fs=args.fs,
            gaussian_prob=args.augment_gaussian_prob,
            gaussian_std_scale=args.augment_gaussian_std_scale,
            bandstop_prob=args.augment_bandstop_prob,
            