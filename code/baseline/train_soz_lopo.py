#!/usr/bin/env python3
"""
Leave-One-Patient-Out (LOPO) training for SOZ localization on the private dataset.

For each patient p:
  - test  = all windows from patient p
  - val   = random `val_frac` of the remaining patients (at least 1 patient)
  - train = the rest

After every fold, predictions for the held-out patient are collected. When all
folds finish, per-channel AUC / seizure-level / patient-level metrics are
computed on the concatenated predictions, so sparse channels (few positive
patients) still receive a meaningful AUC.

Usage:
    python code/baseline/train_soz_lopo.py \
        --private_dir /mnt/hd1/dyf/dataset/process_dataset/baseline_private \
        --output_dir  runs/baseline_lopo
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_CODE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from baseline.model import SimpleCNN
from baseline.train import set_seed, get_device, train_one_epoch, evaluate
from baseline.evaluate import compute_soz_metrics, _print_metrics, TCP_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# File-list dataset (bypasses split directories)
# ──────────────────────────────────────────────────────────────────────────────

class FileListSOZDataset(Dataset):
    """SOZ dataset from an explicit list of .npz files."""

    def __init__(self, files, return_meta=False):
        self.files = list(files)
        self.return_meta = return_meta
        if len(self.files) == 0:
            raise ValueError('FileListSOZDataset got empty file list')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        eeg = torch.from_numpy(d['eeg_data'])
        mask = torch.from_numpy(d['channel_mask'])
        label = torch.from_numpy(d['soz_labels'])
        if self.return_meta:
            pid = str(d['patient_id'])
            return eeg, label, mask, pid, self.files[idx]
        return eeg, label, mask


# ──────────────────────────────────────────────────────────────────────────────
# Data scanning
# ──────────────────────────────────────────────────────────────────────────────

def scan_private_files(private_dir):
    """Collect all .npz files across train/val/test subfolders and group by patient."""
    files = []
    for split in ('train', 'val', 'test'):
        files.extend(glob.glob(os.path.join(private_dir, split, '*.npz')))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f'No .npz files found under {private_dir}')

    by_pt = defaultdict(list)
    for f in files:
        # read patient_id quickly
        d = np.load(f, allow_pickle=True)
        pid = str(d['patient_id'])
        by_pt[pid].append(f)
    return dict(by_pt)


# ──────────────────────────────────────────────────────────────────────────────
# One fold
# ──────────────────────────────────────────────────────────────────────────────

def run_fold(test_pid, by_pt, args, device, fold_dir, rng):
    """Train one LOPO fold; return predictions on test patient."""
    os.makedirs(fold_dir, exist_ok=True)

    all_pts = sorted(by_pt.keys())
    other_pts = [p for p in all_pts if p != test_pid]
    rng.shuffle(other_pts)
    n_val = max(1, int(round(len(other_pts) * args.val_frac)))
    val_pts = other_pts[:n_val]
    train_pts = other_pts[n_val:]

    train_files = [f for p in train_pts for f in by_pt[p]]
    val_files   = [f for p in val_pts   for f in by_pt[p]]
    test_files  = list(by_pt[test_pid])

    train_ds = FileListSOZDataset(train_files)
    val_ds   = FileListSOZDataset(val_files)
    test_ds  = FileListSOZDataset(test_files, return_meta=True)

    print(f'  train={len(train_files)} (pts={len(train_pts)})  '
          f'val={len(val_files)} (pts={len(val_pts)})  '
          f'test={len(test_files)} (pt={test_pid})')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = SimpleCNN(n_channels=22, task='soz').to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )

    best_val = float('inf')
    best_path = os.path.join(fold_dir, 'best_model.pt')
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, 'soz')
        val_m = evaluate(model, val_loader, criterion, device, 'soz')
        scheduler.step(val_m['loss'])
        history.append({'epoch': epoch, 'train_loss': tr_loss,
                        'val_loss': val_m['loss'], 'val_mean_auc': val_m['mean_auc']})
        print(f'    ep{epoch:02d} tr={tr_loss:.4f} val={val_m["loss"]:.4f} '
              f'auc={val_m["mean_auc"]:.4f} ({time.time()-t0:.1f}s)')

        if val_m['loss'] < best_val:
            best_val = val_m['loss']
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f'    early stop @ epoch {epoch}')
                break

    # ── Collect predictions on held-out patient (single-pass + MC dropout) ──
    model.load_state_dict(torch.load(best_path, weights_only=True))

    # single pass
    model.eval()
    sp_probs, labels, pids, fpaths = [], [], [], []
    with torch.no_grad():
        for eeg, lab, mk, pid, fp in test_loader:
            eeg = eeg.to(device)
            p = torch.sigmoid(model(eeg)).cpu().numpy()
            sp_probs.append(p)
            labels.append(lab.numpy())
            pids.extend(pid)
            fpaths.extend(fp)
    sp_probs = np.concatenate(sp_probs)
    labels   = np.concatenate(labels)

    # MC dropout
    mc_probs = None
    if args.mc_samples > 0:
        model.train()  # enable dropout
        mc_probs = [np.zeros((args.mc_samples, 22), dtype=np.float32)
                    for _ in range(len(test_ds))]
        idx = 0
        with torch.no_grad():
            for eeg, lab, mk, pid, fp in test_loader:
                eeg = eeg.to(device)
                B = eeg.shape[0]
                for s in range(args.mc_samples):
                    p = torch.sigmoid(model(eeg)).cpu().numpy()
                    for b in range(B):
                        mc_probs[idx + b][s] = p[b]
                idx += B

    fold_summary = {
        'test_pid': test_pid,
        'best_val_loss': best_val,
        'n_train': len(train_files), 'n_val': len(val_files), 'n_test': len(test_files),
        'history': history,
    }
    with open(os.path.join(fold_dir, 'fold_summary.json'), 'w') as f:
        json.dump(fold_summary, f, indent=2)

    # free GPU before next fold
    del model
    torch.cuda.empty_cache()

    return {
        'sp_probs': sp_probs, 'labels': labels, 'pids': pids, 'fpaths': fpaths,
        'mc_probs': mc_probs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f'Device: {device}')

    print(f'Scanning {args.private_dir} ...')
    by_pt = scan_private_files(args.private_dir)
    patients = sorted(by_pt.keys())
    print(f'Found {len(patients)} patients, '
          f'{sum(len(v) for v in by_pt.values())} windows total')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'soz_lopo_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    all_sp_probs, all_labels, all_pids, all_fpaths = [], [], [], []
    all_mc_probs = []

    for fi, test_pid in enumerate(patients, 1):
        print(f'\n[Fold {fi}/{len(patients)}] held-out patient = {test_pid}')
        fold_dir = os.path.join(run_dir, f'fold_{fi:02d}_{_safe(test_pid)}')
        out = run_fold(test_pid, by_pt, args, device, fold_dir, rng)

        all_sp_probs.append(out['sp_probs'])
        all_labels.append(out['labels'])
        all_pids.extend(out['pids'])
        all_fpaths.extend(out['fpaths'])
        if out['mc_probs'] is not None:
            all_mc_probs.extend(out['mc_probs'])

    sp_probs = np.concatenate(all_sp_probs)
    labels   = np.concatenate(all_labels)

    print(f'\n{"="*60}\n  LOPO aggregated evaluation (N={len(labels)} windows)\n{"="*60}')

    sp = compute_soz_metrics(sp_probs, labels, all_pids, all_fpaths, mc_mode=False)
    _print_metrics('LOPO Single-pass', sp['metrics'])

    final = {'single_pass': sp}
    if all_mc_probs:
        mc = compute_soz_metrics(all_mc_probs, labels, all_pids, all_fpaths, mc_mode=True)
        _print_metrics(f'LOPO MC Dropout (n={args.mc_samples})', mc['metrics'])
        final['mc_dropout'] = mc

    # Save
    out_json = {
        'args': vars(args),
        'n_patients': len(patients),
        'patients': patients,
        'single_pass_metrics': _json(sp['metrics']),
        'mc_dropout_metrics': _json(final['mc_dropout']['metrics']) if 'mc_dropout' in final else None,
    }
    with open(os.path.join(run_dir, 'lopo_results.json'), 'w') as f:
        json.dump(out_json, f, indent=2, default=str)
    # raw predictions for later analysis
    np.savez_compressed(os.path.join(run_dir, 'lopo_predictions.npz'),
                        sp_probs=sp_probs, labels=labels,
                        pids=np.array(all_pids), fpaths=np.array(all_fpaths),
                        mc_probs=np.array(all_mc_probs) if all_mc_probs else np.zeros(0))
    print(f'\nResults saved to {run_dir}')


def _safe(s):
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(s))


def _json(metrics):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer))
                else [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in v]
                if isinstance(v, list) else v)
            for k, v in metrics.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='LOPO SOZ training on private dataset')
    p.add_argument('--private_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='runs/baseline_lopo')
    p.add_argument('--val_frac', type=float, default=0.15,
                   help='Fraction of non-held-out patients used for validation')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mc_samples', type=int, default=20)
    main(p.parse_args())
