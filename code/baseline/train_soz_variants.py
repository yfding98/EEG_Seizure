#!/usr/bin/env python3
"""
Train baseline CNN for SOZ localization with different data sources.

Supports three training modes:
  - private:  Train/val/test all from private data (patient-level split)
  - combined: Train on TUSZ+private, val on TUSZ dev+private val, test on private test
  - tusz:     Train/val/test from TUSZ only (same as train.py --task soz)

Usage:
    # Private data only
    python code/baseline/train_soz_variants.py --source private \
        --private_dir F:/process_dataset/baseline_private

    # TUSZ + private combined
    python code/baseline/train_soz_variants.py --source combined \
        --tusz_dir F:/process_dataset/baseline \
        --private_dir F:/process_dataset/baseline_private

    # TUSZ only (equivalent to train.py --task soz)
    python code/baseline/train_soz_variants.py --source tusz \
        --tusz_dir F:/process_dataset/baseline
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_CODE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from baseline.dataset import TUSZBaselineDataset
from baseline.model import SimpleCNN
from baseline.train import (
    set_seed, get_device, train_one_epoch, evaluate, compute_metrics,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset building
# ──────────────────────────────────────────────────────────────────────────────

def _try_load(data_dir, split, task, return_meta=False):
    """Load dataset, return None if split dir is empty or missing."""
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        return None
    try:
        return TUSZBaselineDataset(data_dir, split, task=task, return_meta=return_meta)
    except FileNotFoundError:
        return None


def build_datasets(source, tusz_dir, private_dir, return_meta=False):
    """
    Build train/val/test datasets according to source mode.

    Returns:
        train_ds, val_ds, test_ds, test_label (str describing test set origin)
    """
    task = 'soz'

    if source == 'private':
        train_ds = _try_load(private_dir, 'train', task, return_meta)
        val_ds   = _try_load(private_dir, 'val',   task, return_meta)
        test_ds  = _try_load(private_dir, 'test',  task, return_meta)
        test_label = 'private-test'

    elif source == 'tusz':
        train_ds = _try_load(tusz_dir, 'train', task, return_meta)
        val_ds   = _try_load(tusz_dir, 'dev',   task, return_meta)
        test_ds  = _try_load(tusz_dir, 'eval',  task, return_meta)
        test_label = 'tusz-eval'

    elif source == 'combined':
        # Train: TUSZ train + private train
        parts_train = []
        t = _try_load(tusz_dir, 'train', task, return_meta)
        if t: parts_train.append(t)
        p = _try_load(private_dir, 'train', task, return_meta)
        if p: parts_train.append(p)
        train_ds = ConcatDataset(parts_train) if len(parts_train) > 1 else (parts_train[0] if parts_train else None)

        # Val: TUSZ dev + private val
        parts_val = []
        t = _try_load(tusz_dir, 'dev', task, return_meta)
        if t: parts_val.append(t)
        p = _try_load(private_dir, 'val', task, return_meta)
        if p: parts_val.append(p)
        val_ds = ConcatDataset(parts_val) if len(parts_val) > 1 else (parts_val[0] if parts_val else None)

        # Test: private test (main interest)
        test_ds = _try_load(private_dir, 'test', task, return_meta)
        test_label = 'private-test'
    else:
        raise ValueError(f'Unknown source: {source}')

    for name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        if ds is None:
            raise FileNotFoundError(f'{name} dataset is empty or missing for source={source}')

    return train_ds, val_ds, test_ds, test_label


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f'Device: {device}')
    print(f'Source: {args.source}')

    # ── Data ──
    train_ds, val_ds, test_ds, test_label = build_datasets(
        args.source, args.tusz_dir, args.private_dir,
    )
    print(f'Train: {len(train_ds)} samples')
    print(f'Val:   {len(val_ds)} samples')
    print(f'Test:  {len(test_ds)} samples  ({test_label})')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model ──
    model = SimpleCNN(n_channels=22, task='soz').to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {param_count:,}')

    # ── Loss ──
    criterion = nn.BCEWithLogitsLoss()

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    )

    # ── Output dir ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'soz_{args.source}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # ── Training loop ──
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    task = 'soz'

    print(f'\n{"="*60}')
    print(f'Training SOZ ({args.source}) for {args.epochs} epochs (patience={args.patience})')
    print(f'{"="*60}\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, task)
        val_metrics = evaluate(model, val_loader, criterion, device, task)
        val_loss = val_metrics['loss']
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'train_loss={train_loss:.4f} | '
              f'val_loss={val_loss:.4f} | '
              f'val_mean_auc={val_metrics["mean_auc"]:.4f} | '
              f'val_f1_macro={val_metrics["f1_macro"]:.4f} | '
              f'{elapsed:.1f}s')

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{k: v for k, v in val_metrics.items() if k != 'per_channel_auc'},
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break

    # ── Test evaluation (basic metrics) ──
    print(f'\n{"="*60}')
    print(f'Test Evaluation on {test_label} (best model)')
    print(f'{"="*60}\n')

    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pt'), weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device, task)

    print(f'  Mean AUC:        {test_metrics["mean_auc"]:.4f}')
    print(f'  F1 (micro):      {test_metrics["f1_micro"]:.4f}')
    print(f'  F1 (macro):      {test_metrics["f1_macro"]:.4f}')
    print(f'  Sample Accuracy: {test_metrics["sample_accuracy"]:.4f}')
    print(f'\n  Per-channel AUC:')
    from baseline.preprocess_tusz import TCP_PAIRS
    for i, (a, c) in enumerate(TCP_PAIRS):
        auc = test_metrics["per_channel_auc"][i]
        print(f'    {a}-{c:3s}: {auc:.4f}')

    # ── DeepSOZ-style evaluation ──
    from baseline.evaluate import evaluate_soz

    # Primary test set
    test_ds_meta = build_datasets(
        args.source, args.tusz_dir, args.private_dir, return_meta=True,
    )[2]
    test_loader_meta = DataLoader(test_ds_meta, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    evaluate_soz(model, test_loader_meta, device,
                 output_dir=run_dir, mc_samples=args.mc_samples)

    # For combined mode: also evaluate on TUSZ eval for cross-dataset comparison
    if args.source == 'combined':
        tusz_eval = _try_load(args.tusz_dir, 'eval', 'soz', return_meta=True)
        if tusz_eval is not None:
            print(f'\n--- Additional: TUSZ eval set ---')
            tusz_eval_loader = DataLoader(tusz_eval, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)
            tusz_eval_dir = os.path.join(run_dir, 'tusz_eval')
            evaluate_soz(model, tusz_eval_loader, device,
                         output_dir=tusz_eval_dir, mc_samples=args.mc_samples)

    # ── Save results ──
    results = {
        'source': args.source,
        'test_set': test_label,
        'args': vars(args),
        'best_val_loss': best_val_loss,
        'test_metrics': {k: v if not isinstance(v, np.floating) else float(v)
                         for k, v in test_metrics.items()},
        'history': history,
        'n_params': param_count,
    }

    results_path = os.path.join(run_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {results_path}')
    print(f'Model saved to {os.path.join(run_dir, "best_model.pt")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='SOZ Baseline - Private / Combined Training')
    p.add_argument('--source', type=str, required=True,
                   choices=['private', 'combined', 'tusz'],
                   help='Data source: private / combined / tusz')
    p.add_argument('--tusz_dir', type=str, default=r'F:\process_dataset\baseline',
                   help='TUSZ preprocessed data directory')
    p.add_argument('--private_dir', type=str, default=r'F:\process_dataset\baseline_private',
                   help='Private preprocessed data directory')
    p.add_argument('--output_dir', type=str, default='runs/baseline')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mc_samples', type=int, default=20,
                   help='MC dropout samples for DeepSOZ evaluation')
    main(p.parse_args())
