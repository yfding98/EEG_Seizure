#!/usr/bin/env python3
"""
Train baseline CNN for seizure detection or SOZ localization.

Usage (run from project root):
    # Seizure detection
    python code/baseline/train.py --task detection --data_dir F:/process_dataset/baseline

    # SOZ localization
    python code/baseline/train.py --task soz --data_dir F:/process_dataset/baseline

    # With custom hyperparams
    python code/baseline/train.py --task detection --lr 1e-4 --epochs 50 --batch_size 32
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# add code/ to sys.path so sibling packages resolve
_CODE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

from baseline.dataset import TUSZBaselineDataset
from baseline.model import (
    MonopolarEEGNetRegionCNN,
    MonopolarRegionCNN,
    MonopolarSeparableRegionCNN,
    MonopolarSharedAttentionRegionCNN,
    RegionCNN,
    SimpleCNN,
)
from baseline.regions import MONOPOLAR_CHANNELS_TUSZ17, REGION_NAMES

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_region_model(input_mode='bipolar', region_model='standard'):
    if input_mode == 'bipolar':
        if region_model != 'standard':
            raise ValueError('Bipolar region mode currently supports only region_model=standard')
        return RegionCNN()

    if region_model == 'standard':
        return MonopolarRegionCNN()
    if region_model == 'separable':
        return MonopolarSeparableRegionCNN()
    if region_model == 'shared_attention':
        return MonopolarSharedAttentionRegionCNN(n_channels=len(MONOPOLAR_CHANNELS_TUSZ17))
    if region_model == 'eegnet':
        return MonopolarEEGNetRegionCNN()
    raise ValueError(f'Unknown region_model: {region_model}')


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    total_loss = 0.0
    n = 0

    for eeg, label, mask in loader:
        eeg = eeg.to(device)
        label = label.to(device)

        logits = model(eeg)

        if task == 'detection':
            loss = criterion(logits.squeeze(-1), label)
        else:
            loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * eeg.size(0)
        n += eeg.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for eeg, label, mask in loader:
        eeg = eeg.to(device)
        label = label.to(device)

        logits = model(eeg)

        if task == 'detection':
            loss = criterion(logits.squeeze(-1), label)
            probs = torch.sigmoid(logits.squeeze(-1))
        else:
            loss = criterion(logits, label)
            probs = torch.sigmoid(logits)

        total_loss += loss.item() * eeg.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    avg_loss = total_loss / sum(len(p) for p in all_probs)
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(probs, labels, task)
    metrics['loss'] = avg_loss
    return metrics


def compute_metrics(probs, labels, task):
    if task == 'detection':
        preds = (probs >= 0.5).astype(int)
        labels_int = labels.astype(int)
        m = {
            'accuracy': accuracy_score(labels_int, preds),
            'f1': f1_score(labels_int, preds, zero_division=0),
            'precision': precision_score(labels_int, preds, zero_division=0),
            'recall': recall_score(labels_int, preds, zero_division=0),
        }
        try:
            m['auc'] = roc_auc_score(labels_int, probs)
        except ValueError:
            m['auc'] = 0.0
        tn, fp, fn, tp = confusion_matrix(labels_int, preds, labels=[0, 1]).ravel()
        m['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        m['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return m
    else:
        # SOZ: per-channel or per-region multi-label metrics
        preds = (probs >= 0.5).astype(int)
        labels_int = labels.astype(int)
        n_outputs = probs.shape[1]

        per_output_auc = []
        for ch in range(n_outputs):
            try:
                auc = roc_auc_score(labels_int[:, ch], probs[:, ch])
            except ValueError:
                auc = 0.0
            per_output_auc.append(auc)

        m = {
            'mean_auc': np.mean(per_output_auc),
            'per_channel_auc': per_output_auc,
            'f1_micro': f1_score(labels_int.ravel(), preds.ravel(), zero_division=0),
            'f1_macro': f1_score(labels_int, preds, average='macro', zero_division=0),
            'accuracy': accuracy_score(labels_int.ravel(), preds.ravel()),
        }

        # sample-level: any SOZ correctly identified
        sample_pred = preds.max(axis=1)
        sample_true = labels_int.max(axis=1)
        m['sample_accuracy'] = accuracy_score(sample_true, sample_pred)
        return m


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f'Device: {device}')
    print(f'Task: {args.task}')
    if args.input_mode == 'monopolar' and args.task != 'soz_region':
        raise ValueError('input_mode=monopolar is currently implemented for --task soz_region only')

    # ── Data ──
    train_ds = TUSZBaselineDataset(
        args.data_dir, 'train', task=args.task, input_mode=args.input_mode,
        allow_bipolar_fallback=args.allow_bipolar_fallback,
    )
    val_ds = TUSZBaselineDataset(
        args.data_dir, 'dev', task=args.task, input_mode=args.input_mode,
        allow_bipolar_fallback=args.allow_bipolar_fallback,
    )
    test_ds = TUSZBaselineDataset(
        args.data_dir, 'eval', task=args.task, input_mode=args.input_mode,
        allow_bipolar_fallback=args.allow_bipolar_fallback,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model ──
    if args.task == 'soz_region':
        model = build_region_model(args.input_mode, args.region_model).to(device)
        print(f'Region outputs: {", ".join(REGION_NAMES)}')
        print(f'Input mode: {args.input_mode}, region_model: {args.region_model}')
        if args.input_mode == 'monopolar':
            print(f'Monopolar inputs: {", ".join(MONOPOLAR_CHANNELS_TUSZ17)}')
    else:
        model = SimpleCNN(n_channels=22, task=args.task).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {param_count:,}')

    # ── Loss ──
    if args.task == 'detection':
        # compute class weight for imbalanced data
        n_pos = sum(1 for f in train_ds.files if 'seizure' in os.path.basename(f)
                    and 'nonseizure' not in os.path.basename(f))
        n_neg = len(train_ds) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f'Class balance: {n_pos} pos / {n_neg} neg, pos_weight={pos_weight.item():.2f}')
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    )

    # ── Output dir ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'{args.task}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # ── Training loop ──
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    print(f'\n{"="*60}')
    print(f'Training for {args.epochs} epochs (patience={args.patience})')
    print(f'{"="*60}\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.task)
        val_metrics = evaluate(model, val_loader, criterion, device, args.task)
        val_loss = val_metrics['loss']
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        if args.task == 'detection':
            print(f'Epoch {epoch:3d}/{args.epochs} | '
                  f'train_loss={train_loss:.4f} | '
                  f'val_loss={val_loss:.4f} | '
                  f'val_acc={val_metrics["accuracy"]:.4f} | '
                  f'val_f1={val_metrics["f1"]:.4f} | '
                  f'val_auc={val_metrics["auc"]:.4f} | '
                  f'{elapsed:.1f}s')
        else:
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

    # ── Test evaluation ──
    print(f'\n{"="*60}')
    print('Test Evaluation (best model)')
    print(f'{"="*60}\n')

    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pt'), weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device, args.task)

    if args.task == 'detection':
        print(f'  Accuracy:    {test_metrics["accuracy"]:.4f}')
        print(f'  F1 Score:    {test_metrics["f1"]:.4f}')
        print(f'  AUC-ROC:     {test_metrics["auc"]:.4f}')
        print(f'  Sensitivity: {test_metrics["sensitivity"]:.4f}')
        print(f'  Specificity: {test_metrics["specificity"]:.4f}')
        print(f'  Precision:   {test_metrics["precision"]:.4f}')
        print(f'  Recall:      {test_metrics["recall"]:.4f}')
    else:
        print(f'  Mean AUC:        {test_metrics["mean_auc"]:.4f}')
        print(f'  F1 (micro):      {test_metrics["f1_micro"]:.4f}')
        print(f'  F1 (macro):      {test_metrics["f1_macro"]:.4f}')
        print(f'  Sample Accuracy: {test_metrics["sample_accuracy"]:.4f}')
        if args.task == 'soz_region':
            print(f'\n  Per-region AUC:')
            for i, name in enumerate(REGION_NAMES):
                auc = test_metrics["per_channel_auc"][i]
                print(f'    {name:15s}: {auc:.4f}')
        else:
            print(f'\n  Per-channel AUC:')
            from baseline.preprocess_tusz import TCP_PAIRS
            for i, (a, c) in enumerate(TCP_PAIRS):
                auc = test_metrics["per_channel_auc"][i]
                print(f'    {a}-{c:3s}: {auc:.4f}')

    # ── DeepSOZ-style evaluation for SOZ task ──
    if args.task == 'soz':
        from baseline.evaluate import evaluate_soz
        eval_ds = TUSZBaselineDataset(
            args.data_dir, 'eval', task='soz', return_meta=True,
        )
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        evaluate_soz(model, eval_loader, device,
                     output_dir=run_dir, mc_samples=args.mc_samples)

    # ── Save results ──
    results = {
        'task': args.task,
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
    p = argparse.ArgumentParser(description='Baseline CNN Training')
    p.add_argument('--task', type=str, default='detection',
                   choices=['detection', 'soz', 'soz_region'])
    p.add_argument('--data_dir', type=str, default=r'F:\process_dataset\baseline')
    p.add_argument('--output_dir', type=str, default='runs/baseline')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mc_samples', type=int, default=20,
                   help='MC dropout samples for SOZ evaluation (default: 20)')
    p.add_argument('--input_mode', type=str, default='bipolar',
                   choices=['bipolar', 'monopolar'],
                   help='Input channel space for region SOZ models')
    p.add_argument('--region_model', type=str, default='standard',
                   choices=['standard', 'separable', 'shared_attention', 'eegnet'],
                   help='Region SOZ model variant')
    p.add_argument('--allow_bipolar_fallback', action='store_true',
                   help='When input_mode=monopolar, approximate monopolar signals from 22 bipolar eeg_data if no monopolar_data exists')
    main(p.parse_args())
