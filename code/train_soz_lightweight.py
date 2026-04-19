#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_soz_lightweight.py

SOZ localization training script using the lightweight 2-layer Transformer
backbone (integration_model_v2) instead of the pretrained LaBraM.

Differences from train_soz_locator_with_brain_networks.py:
  - No LaBraM checkpoint loading
  - No stage-1 pretraining
  - Simplified training: all parameters trainable from the start
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
        # metrics
        build_selection_key, format_selection_key_text,
        parse_brain_network_features,
        load_compatible_model_weights,
        SUPPORTED_BRAIN_NETWORK_FEATURES,
    )
    from models.manifest_dataset import get_region_names
    from models.region_confusion import save_region_confusion_report
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
        build_selection_key, format_selection_key_text,
        parse_brain_network_features,
        load_compatible_model_weights,
        SUPPORTED_BRAIN_NETWORK_FEATURES,
    )
    from .manifest_dataset import get_region_names
    from .region_confusion import save_region_confusion_report
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
# Argument Parser (simplified — no LaBraM / stage pretraining args)
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

    # brain networks
    p.add_argument('--brain-network-features', default='gc,te,aec,wpli')

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
# Main
# =====================================================================

def main():
    args = parse_args()
    selected_brain_features = parse_brain_network_features(args.brain_network_features)

    if args.init_ckpt and args.resume:
        raise ValueError("--init-ckpt and --resume cannot be used together")

    rank, world, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir)
    if is_main(rank):
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

    patch_len = int(args.patch_duration * args.fs)
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
            bandstop_min_freq=args.augment_bandstop_min_freq,
            bandstop_max_freq=args.augment_bandstop_max_freq,
            bandstop_width_hz=args.augment_bandstop_width_hz,
            channel_dropout_prob=args.augment_channel_drop_prob,
            max_channel_drops=args.augment_max_channel_drops,
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
        if args.augment_minority_oversample > 0.0:
            minority_oversampler = MinorityClassOversampler(
                n_segments=args.augment_sr_segments,
                oversample_ratio=args.augment_minority_oversample,
                region_negative_threshold=args.augment_minority_region_threshold,
                augmentor=train_augmentor,
            )
        log.info("  EEG augmentation enabled")

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

    # ── 2. Model init ──
    log.info("=== Step 2: Initializing model ===")
    region_names = tuple(split_meta.get('region_names', get_region_names(args.region_label_mode)))
    n_hemisphere_classes = 2 if split_meta.get('hemisphere_label_mode') == 'lr_ignore_b' else 3

    cfg = IntegrationConfig(
        task_mode='soz',
        embed_dim=args.embed_dim,
        n_transformer_layers=args.n_transformer_layers,
        n_heads_transformer=args.n_heads,
        ff_mult=args.ff_mult,
        transformer_dropout=args.transformer_dropout,
        patch_len=patch_len,
        n_pre_patches=n_pre_patches,
        n_post_patches=n_post_patches,
        fs=args.fs,
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
    model = Lightweight_Transformer_BrainNetwork_Integration(cfg).to(device)
    log.info(model.summary())

    # Load init checkpoint if provided
    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_path}")
        load_info = load_compatible_model_weights(model, str(init_path), map_location=device)
        log.info(
            "  Loaded init checkpoint: loaded=%d missing=%d unexpected=%d",
            len(load_info['loaded_keys']),
            len(load_info['missing_keys']),
            len(load_info['unexpected_keys']),
        )

    # Set pos_weight
    log.info("  Computing pos_weight ...")
    if train_analysis is not None:
        pw = compute_pos_weight_from_analysis(train_analysis, device=device)
    else:
        pw = compute_pos_weight(train_loader, device=device)
    model.set_pos_weight(pw)

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
            log.info("  Private channel loss weighting enabled")

    # ── 3. Training ──
    log.info("=== Step 3: Training ===")
    writer = SummaryWriter(str(output_dir / 'tb')) if (_HAS_TB and is_main(rank)) else None
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base_model = model.module if hasattr(model, 'module') else model

    # Resume
    start_epoch = 0
    best_selection_key = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        raw_key = ckpt.get('best_selection_key', None)
        if raw_key is not None:
            best_selection_key = tuple(float(x) for x in raw_key)
            if len(best_selection_key) < 6:
                best_selection_key = best_selection_key + (-1.0,) * (6 - len(best_selection_key))
        log.info("  Resumed from epoch %d", start_epoch)

    total_epochs = args.epochs

    # All params trainable from start — no frozen backbone phases
    optimizer = torch.optim.AdamW(
        base_model.get_param_groups(args.lr), weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=total_epochs,
    )
    # Advance scheduler to resume point
    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, total_epochs):
        if world > 1:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, cfg, writer,
            generalized_pos_ratio_threshold=args.generalized_pos_ratio_threshold,
            generalized_sample_weight=args.generalized_sample_weight,
            train_augmentor=train_augmentor,
            lr_mirror_prob=train_lr_mirror_prob,
            minority_oversampler=minority_oversampler,
        )
        scheduler.step()
        dt = time.time() - t0

        val_metrics = evaluate(
            model, val_loader, device,
            generalized_pos_ratio_threshold=args.generalized_pos_ratio_threshold,
            generalized_sample_weight=args.generalized_sample_weight,
            collect_sample_info=True,
        )

        # DeepSOZ SOZ metrics (lightweight per-epoch)
        soz_deepsoz: Dict[str, float] = {}
        if is_main(rank) and 'patient_ids' in val_metrics:
            try:
                soz_deepsoz = compute_deepsoz_soz_metrics(
                    probs=val_metrics['probs'],
                    targets=val_metrics['targets'],
                    patient_ids=val_metrics['patient_ids'],
                    edf_paths=val_metrics['edf_paths'],
                    neighbour_threshold=args.neighbour_threshold,
                )
            except Exception as exc:
                log.warning("DeepSOZ SOZ metrics failed at epoch %d: %s", epoch + 1, exc)

        if is_main(rank):
            current_lr = optimizer.param_groups[-1]['lr']
            log.info(
                f"Epoch {epoch:3d}/{total_epochs} "
                f"loss={train_metrics['loss']:.4f} "
                f"train_r3={train_metrics['recall_at_3']:.3f} "
                f"train_ndcg3={train_metrics['ndcg_at_3']:.3f} "
                f"val_r3={val_metrics['recall_at_3']:.3f} "
                f"val_ndcg3={val_metrics['ndcg_at_3']:.3f} "
                f"val_auc={val_metrics['auc']:.3f} "
                f"val_region={val_metrics['region_acc']:.3f} "
                f"val_hemi={val_metrics['hemisphere_acc']:.3f} "
                f"lr={current_lr:.2e} ({dt:.1f}s)"
            )
            if writer:
                writer.add_scalar('val/recall_at_3', val_metrics['recall_at_3'], epoch)
                writer.add_scalar('val/ndcg_at_3', val_metrics['ndcg_at_3'], epoch)
                writer.add_scalar('val/mrr', val_metrics['mrr'], epoch)
                writer.add_scalar('val/auc', val_metrics['auc'], epoch)
                writer.add_scalar('val/region_acc', val_metrics['region_acc'], epoch)
                writer.add_scalar('val/hemisphere_acc', val_metrics['hemisphere_acc'], epoch)
                writer.add_scalar('lr', current_lr, epoch)
                if soz_deepsoz:
                    for dkey, dval in soz_deepsoz.items():
                        if isinstance(dval, (int, float)):
                            writer.add_scalar(f'val_deepsoz_soz/{dkey}', dval, epoch)

            if soz_deepsoz:
                log.info(
                    "  [soz deepsoz] corr_sz=%.3f acc_pt_w=%.3f acc_pt_s=%.3f",
                    soz_deepsoz.get('corr_sz', 0.0),
                    soz_deepsoz.get('acc_pt_weighted', 0.0),
                    soz_deepsoz.get('acc_pt_strict', 0.0),
                )

            # Save best
            val_selection_key = build_selection_key(val_metrics, args.task_training_mode)
            if val_selection_key > best_selection_key:
                best_selection_key = val_selection_key
                val_summary = {
                    key: value for key, value in val_metrics.items()
                    if not isinstance(value, np.ndarray)
                }
                base_model.save_checkpoint(
                    str(output_dir / 'best_model.pt'),
                    extra={
                        'epoch': epoch,
                        'best_selection_key': list(best_selection_key),
                        'val_metrics': val_summary,
                        'deepsoz_soz_metrics': soz_deepsoz,
                    },
                )
                log.info(
                    "  ** New best: %s",
                    format_selection_key_text(best_selection_key, args.task_training_mode),
                )

            if (epoch + 1) % args.save_every == 0:
                base_model.save_checkpoint(
                    str(output_dir / f'ckpt_epoch{epoch:03d}.pt'),
                    extra={
                        'epoch': epoch,
                        'best_selection_key': list(best_selection_key),
                    },
                )

    # ── 4. Test evaluation ──
    log.info("=== Step 4: Test evaluation ===")
    best_path = output_dir / 'best_model.pt'
    if best_path.exists():
        ckpt_best = torch.load(str(best_path), map_location=device)
        base_model.load_state_dict(ckpt_best['model_state'])

    val_metrics_best = evaluate(
        model, val_loader, device,
        generalized_pos_ratio_threshold=args.generalized_pos_ratio_threshold,
        generalized_sample_weight=args.generalized_sample_weight,
    )
    test_metrics = evaluate(
        model, test_loader, device,
        generalized_pos_ratio_threshold=args.generalized_pos_ratio_threshold,
        generalized_sample_weight=args.generalized_sample_weight,
        collect_sample_info=True,
    )

    # MC dropout DeepSOZ SOZ evaluation
    mc_soz_results: Dict[str, object] = {}
    if is_main(rank):
        log.info(
            "Running MC dropout SOZ evaluation (mc_samples=%d) ...",
            args.mc_samples,
        )
        try:
            mc_soz_results = run_detailed_soz_evaluation(
                model, test_loader, device,
                mc_samples=args.mc_samples,
                neighbour_threshold=args.neighbour_threshold,
                output_dir=str(output_dir),
            )
            mc_m = mc_soz_results.get('metrics', {})
            log.info(
                "  [MC soz] corr_sz=%.3f acc_pt_w=%.3f acc_pt_s=%.3f",
                mc_m.get('corr_sz', 0.0),
                mc_m.get('acc_pt_weighted', 0.0),
                mc_m.get('acc_pt_strict', 0.0),
            )
        except Exception as exc:
            log.warning("MC dropout SOZ evaluation failed: %s", exc)

    if is_main(rank):
        log.info(
            f"\nTest results:\n"
            f"  Recall@1: {test_metrics['recall_at_1']:.4f}\n"
            f"  Recall@3: {test_metrics['recall_at_3']:.4f}\n"
            f"  Recall@5: {test_metrics['recall_at_5']:.4f}\n"
            f"  nDCG@3:   {test_metrics['ndcg_at_3']:.4f}\n"
            f"  MRR:      {test_metrics['mrr']:.4f}\n"
            f"  AUC:      {test_metrics['auc']:.4f}\n"
            f"  Region:   {test_metrics['region_acc']:.4f}\n"
            f"  Hemi:     {test_metrics['hemisphere_acc']:.4f}"
        )

        # Report
        report = (
            f"# SOZ Localization Report (Lightweight Transformer v2)\n\n"
            f"## Model\n"
            f"- Backbone: {args.n_transformer_layers}-layer Transformer "
            f"(embed_dim={args.embed_dim}, heads={args.n_heads})\n"
            f"- No pretrained LaBraM weights\n"
            f"- Training mode: {args.task_training_mode}\n"
            f"- Epochs: {total_epochs}\n"
            f"- Output mode: {args.output_mode}\n"
            f"- Region label mode: {args.region_label_mode}\n\n"
            f"## Test Metrics\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Recall@1 | {test_metrics['recall_at_1']:.4f} |\n"
            f"| Recall@3 | {test_metrics['recall_at_3']:.4f} |\n"
            f"| Recall@5 | {test_metrics['recall_at_5']:.4f} |\n"
            f"| nDCG@3 | {test_metrics['ndcg_at_3']:.4f} |\n"
            f"| MRR | {test_metrics['mrr']:.4f} |\n"
            f"| AUC | {test_metrics['auc']:.4f} |\n"
            f"| Region acc | {test_metrics['region_acc']:.4f} |\n"
            f"| Hemisphere acc | {test_metrics['hemisphere_acc']:.4f} |\n\n"
            f"## Best validation key\n"
            f"{format_selection_key_text(best_selection_key, args.task_training_mode)}\n"
        )
        if mc_soz_results and 'metrics' in mc_soz_results:
            mc_m = mc_soz_results['metrics']
            report += (
                f"\n## DeepSOZ SOZ (MC dropout, N={args.mc_samples})\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| corr_sz | {mc_m.get('corr_sz', 0.0):.4f} |\n"
                f"| acc_pt (weighted) | {mc_m.get('acc_pt_weighted', 0.0):.4f} |\n"
                f"| acc_pt (strict) | {mc_m.get('acc_pt_strict', 0.0):.4f} |\n"
                f"| acc_pt (lenient) | {mc_m.get('acc_pt_lenient', 0.0):.4f} |\n"
            )
        (output_dir / 'report.md').write_text(report, encoding='utf-8')
        log.info("Report saved to %s", output_dir / 'report.md')

        # Save predictions
        np.savez(
            str(output_dir / 'val_predictions.npz'),
            probs=val_metrics_best['probs'],
            targets=val_metrics_best['targets'],
            region_probs=val_metrics_best['region_probs'],
            region_targets=val_metrics_best['region_targets'],
            region_names=np.asarray(region_names),
            hemisphere_logits=val_metrics_best['hemisphere_logits'],
            hemisphere_targets=val_metrics_best['hemisphere_targets'],
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
        )
        save_region_confusion_report(
            region_probs=test_metrics['region_probs'],
            region_targets=test_metrics['region_targets'],
            output_dir=output_dir,
            threshold=0.5,
            region_names=region_names,
        )

    if writer:
        writer.close()
    if world > 1:
        dist.destroy_process_group()

    log.info("Done.")
    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    sys.exit(main())
