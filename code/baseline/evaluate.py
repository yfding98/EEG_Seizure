#!/usr/bin/env python3
"""
DeepSOZ-style evaluation for baseline CNN SOZ localization.

Implements the evaluation pipeline from:
    Narasimhan et al., "DeepSOZ: A Robust Deep Model for Joint Temporal and
    Spatial Seizure Onset Localization from Multichannel EEG Data"

Evaluation hierarchy:
    1. Per-channel AUC (standard ML metric)
    2. Seizure-level SOZ accuracy (argmax + neighborhood relaxation)
    3. Patient-level SOZ accuracy (aggregate across seizures → argmax)
    4. MC dropout uncertainty estimation

Usage (standalone):
    python code/baseline/evaluate.py \
        --model_path runs/baseline/soz_xxx/best_model.pt \
        --data_dir F:/process_dataset/baseline \
        --mc_samples 20

Usage (from training script):
    from baseline.evaluate import evaluate_soz
    results = evaluate_soz(model, test_loader, device, output_dir='runs/...')
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from baseline.dataset import TUSZBaselineDataset
from baseline.model import SimpleCNN
from baseline.preprocess_tusz import TCP_PAIRS

# ──────────────────────────────────────────────────────────────────────────────
# 22-channel bipolar channel definitions
# ──────────────────────────────────────────────────────────────────────────────

TCP_NAMES = [f'{a}-{b}' for a, b in TCP_PAIRS]

# Spatial neighbors: two bipolar channels are neighbors if they share an electrode
# or are adjacent in the same chain.
def _build_bipolar_neighbors():
    neighbors = defaultdict(set)
    # within-chain adjacency
    chains = [
        [0, 1, 2, 3],          # left temporal
        [4, 5, 6, 7],          # right temporal
        [8, 9, 10, 11, 12, 13],  # central
        [14, 15, 16, 17],      # left parasagittal
        [18, 19, 20, 21],      # right parasagittal
    ]
    for chain in chains:
        for i in range(len(chain) - 1):
            neighbors[chain[i]].add(chain[i + 1])
            neighbors[chain[i + 1]].add(chain[i])

    # cross-chain: channels sharing a monopolar electrode
    for i in range(22):
        elecs_i = set(TCP_PAIRS[i])
        for j in range(i + 1, 22):
            elecs_j = set(TCP_PAIRS[j])
            if elecs_i & elecs_j:
                neighbors[i].add(j)
                neighbors[j].add(i)

    return dict(neighbors)


BIPOLAR_NEIGHBORS = _build_bipolar_neighbors()


# ──────────────────────────────────────────────────────────────────────────────
# DeepSOZ core functions
# ──────────────────────────────────────────────────────────────────────────────

def check_neighborhood(max_chn, onset_map):
    """Check if predicted channel is a spatial neighbor of any true SOZ channel."""
    for i in range(len(onset_map)):
        if onset_map[i] == 1:
            if max_chn in BIPOLAR_NEIGHBORS.get(i, set()):
                return True
    return False


def final_loc(psoz, true_onset, neighbour_threshold=4):
    """
    DeepSOZ-style SOZ localization evaluation.

    Args:
        psoz:       (N, 22) - N predictions (MC samples or seizures), raw probabilities
        true_onset: (22,)   - binary ground-truth SOZ labels
        neighbour_threshold: max true SOZ channels for neighborhood relaxation

    Returns:
        ysoz:        (22,) - averaged normalized SOZ scores
        uncertainty: (22,) - per-channel variance across samples
        correct:     int   - 1 if correctly localized, 0 otherwise
    """
    if psoz.ndim == 1:
        psoz = psoz[np.newaxis, :]

    # row-wise max-normalization
    row_max = psoz.max(axis=1, keepdims=True)
    row_max = np.clip(row_max, 1e-8, None)
    psoz_norm = psoz / row_max

    # average across samples
    ysoz = psoz_norm.mean(axis=0)

    # argmax prediction
    max_chn = int(np.argmax(ysoz))
    correct = 1 if true_onset[max_chn] == 1 else 0

    # neighborhood relaxation
    if correct == 0 and true_onset.sum() <= neighbour_threshold:
        if check_neighborhood(max_chn, true_onset):
            correct = 1

    # uncertainty (variance across MC samples)
    if psoz.shape[0] > 1:
        uncertainty = np.nan_to_num(psoz_norm.var(axis=0))
    else:
        uncertainty = np.zeros(psoz.shape[1])

    return ysoz, uncertainty, correct


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _single_pass(model, loader, device):
    """Single forward pass. Returns probs, labels, patient_ids, file_paths."""
    model.eval()
    all_probs, all_labels, all_pids, all_paths = [], [], [], []

    for batch in loader:
        eeg, label, mask, pid, fpath = batch
        eeg = eeg.to(device)
        logits = model(eeg)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label.numpy())
        all_pids.extend(pid)
        all_paths.extend(fpath)

    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            all_pids, all_paths)


def _mc_inference(model, loader, device, mc_samples=20):
    """MC dropout inference. Returns list of (mc_samples, 22) per sample."""
    model.train()  # keep dropout active

    # collect single-pass data first to get labels/pids
    all_labels, all_pids, all_paths = [], [], []
    all_eeg = []

    for batch in loader:
        eeg, label, mask, pid, fpath = batch
        all_eeg.append(eeg)
        all_labels.append(label.numpy())
        all_pids.extend(pid)
        all_paths.extend(fpath)

    labels = np.concatenate(all_labels)

    # MC forward passes
    mc_probs = []  # list of (mc_samples, 22) per sample
    with torch.no_grad():
        for eeg_batch in all_eeg:
            eeg_batch = eeg_batch.to(device)
            batch_mc = []
            for _ in range(mc_samples):
                logits = model(eeg_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                batch_mc.append(probs)
            # batch_mc: list of mc_samples arrays, each (B, 22)
            batch_mc = np.stack(batch_mc, axis=1)  # (B, mc_samples, 22)
            for i in range(batch_mc.shape[0]):
                mc_probs.append(batch_mc[i])  # (mc_samples, 22)

    return mc_probs, labels, all_pids, all_paths


# ──────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────────────

def _per_channel_auc(probs, labels):
    """Compute per-channel AUC-ROC."""
    n_ch = probs.shape[1]
    aucs = []
    for ch in range(n_ch):
        try:
            auc = roc_auc_score(labels[:, ch], probs[:, ch])
        except ValueError:
            auc = float('nan')
        aucs.append(auc)
    return aucs


def _onset_key(onset):
    """Serialize binary onset map as hashable key for patient-level grouping."""
    return ','.join(str(int(x)) for x in onset)


def compute_soz_metrics(probs_or_mc, labels, patient_ids, file_paths,
                        mc_mode=False, neighbour_threshold=4):
    """
    Compute DeepSOZ-style SOZ metrics at seizure and patient level.

    Args:
        probs_or_mc: if mc_mode=False: (N, 22) probabilities
                     if mc_mode=True:  list of N arrays, each (mc_samples, 22)
        labels:      (N, 22) binary ground-truth
        patient_ids: list of N patient IDs
        file_paths:  list of N file paths
        mc_mode:     whether probs_or_mc contains MC samples
        neighbour_threshold: for neighborhood relaxation

    Returns:
        dict with metrics and per-seizure/per-patient details
    """
    n_samples = len(patient_ids)

    # ── Per-channel AUC ──
    if mc_mode:
        probs_mean = np.array([p.mean(axis=0) for p in probs_or_mc])
    else:
        probs_mean = probs_or_mc
    ch_aucs = _per_channel_auc(probs_mean, labels)

    # ── Seizure-level ──
    per_seizure = []
    for i in range(n_samples):
        true_onset = labels[i]
        if true_onset.sum() == 0:
            continue

        if mc_mode:
            psoz = probs_or_mc[i]  # (mc_samples, 22)
        else:
            psoz = probs_mean[i:i+1]  # (1, 22)

        ysoz, unc, correct = final_loc(psoz, true_onset, neighbour_threshold)

        soz_channels = [TCP_NAMES[j] for j in range(22) if true_onset[j] == 1]
        pred_chn = int(np.argmax(ysoz))

        per_seizure.append({
            'patient_id': patient_ids[i],
            'file_path': os.path.basename(file_paths[i]),
            'correct': correct,
            'pred_channel': TCP_NAMES[pred_chn],
            'pred_channel_idx': pred_chn,
            'true_soz': soz_channels,
            'unc_max': float(unc.max()),
        })

    corr_sz = np.mean([s['correct'] for s in per_seizure]) if per_seizure else 0.0
    szunc_mean = np.mean([s['unc_max'] for s in per_seizure]) if per_seizure else 0.0

    # ── Patient-level ──
    # Group by patient, then by SOZ pattern
    patient_groups = defaultdict(lambda: defaultdict(list))
    for i in range(n_samples):
        true_onset = labels[i]
        if true_onset.sum() == 0:
            continue
        pid = patient_ids[i]
        key = _onset_key(true_onset)
        if mc_mode:
            patient_groups[pid][key].append(probs_or_mc[i])
        else:
            patient_groups[pid][key].append(probs_mean[i])

    per_patient = []
    for pid, patterns in patient_groups.items():
        pattern_results = []
        for onset_key, pred_list in patterns.items():
            true_onset = np.array([int(x) for x in onset_key.split(',')])

            if mc_mode:
                # concatenate all MC samples from all seizures
                stacked = np.concatenate(pred_list, axis=0)  # (N*mc, 22)
            else:
                stacked = np.stack(pred_list, axis=0)  # (N_seizures, 22)

            ysoz, unc, correct = final_loc(stacked, true_onset, neighbour_threshold)
            pattern_results.append({
                'onset_key': onset_key,
                'correct': correct,
                'n_seizures': len(pred_list),
            })

        n_total_sz = sum(p['n_seizures'] for p in pattern_results)
        weighted_score = sum(
            p['correct'] * p['n_seizures'] / n_total_sz
            for p in pattern_results
        )
        correct_all = int(all(p['correct'] for p in pattern_results))
        correct_any = int(any(p['correct'] for p in pattern_results))
        correct_weighted = int(weighted_score >= 0.5)

        soz_chs = set()
        for onset_key in patterns:
            onset = [int(x) for x in onset_key.split(',')]
            for j, v in enumerate(onset):
                if v == 1:
                    soz_chs.add(TCP_NAMES[j])

        per_patient.append({
            'patient_id': pid,
            'correct_weighted': correct_weighted,
            'correct_strict': correct_all,
            'correct_lenient': correct_any,
            'weighted_score': float(weighted_score),
            'n_seizures': n_total_sz,
            'n_patterns': len(pattern_results),
            'soz_channels': sorted(soz_chs),
        })

    n_patients = len(per_patient)
    acc_pt_weighted = np.mean([p['correct_weighted'] for p in per_patient]) if per_patient else 0.0
    acc_pt_strict   = np.mean([p['correct_strict'] for p in per_patient]) if per_patient else 0.0
    acc_pt_lenient  = np.mean([p['correct_lenient'] for p in per_patient]) if per_patient else 0.0

    metrics = {
        'per_channel_auc': ch_aucs,
        'mean_auc': float(np.nanmean(ch_aucs)),
        'corr_sz': float(corr_sz),
        'szunc_mean': float(szunc_mean),
        'acc_pt_weighted': float(acc_pt_weighted),
        'acc_pt_strict': float(acc_pt_strict),
        'acc_pt_lenient': float(acc_pt_lenient),
        'n_seizures_evaluated': len(per_seizure),
        'n_patients_evaluated': n_patients,
    }

    return {
        'metrics': metrics,
        'per_seizure': per_seizure,
        'per_patient': per_patient,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation entry point
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_soz(model, test_loader, device, output_dir=None,
                 mc_samples=20, neighbour_threshold=4):
    """
    Full DeepSOZ-style evaluation for SOZ localization.

    Args:
        model:       trained SimpleCNN (task='soz')
        test_loader: DataLoader with return_meta=True
        device:      torch device
        output_dir:  if set, save results to this directory
        mc_samples:  number of MC dropout forward passes (0 = single pass only)
        neighbour_threshold: neighborhood relaxation threshold

    Returns:
        dict with 'single_pass' and optionally 'mc_dropout' results
    """
    results = {}

    # ── Single-pass evaluation ──
    print('\n[Eval] Running single-pass inference...')
    probs, labels, pids, fpaths = _single_pass(model, test_loader, device)
    sp_results = compute_soz_metrics(
        probs, labels, pids, fpaths,
        mc_mode=False, neighbour_threshold=neighbour_threshold,
    )
    results['single_pass'] = sp_results

    _print_metrics('Single-pass', sp_results['metrics'])

    # ── MC dropout evaluation ──
    if mc_samples > 0:
        print(f'\n[Eval] Running MC dropout inference ({mc_samples} samples)...')
        mc_probs, labels, pids, fpaths = _mc_inference(
            model, test_loader, device, mc_samples,
        )
        mc_results = compute_soz_metrics(
            mc_probs, labels, pids, fpaths,
            mc_mode=True, neighbour_threshold=neighbour_threshold,
        )
        results['mc_dropout'] = mc_results
        _print_metrics(f'MC Dropout (n={mc_samples})', mc_results['metrics'])

    # ── Save results ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _save_results(results, output_dir)

    return results


def _print_metrics(title, metrics):
    """Print evaluation metrics."""
    print(f'\n{"="*60}')
    print(f'  {title} SOZ Evaluation')
    print(f'{"="*60}')
    print(f'  Mean AUC (per-channel):  {metrics["mean_auc"]:.4f}')
    print(f'  Seizure-level accuracy:  {metrics["corr_sz"]:.4f}  '
          f'({metrics["n_seizures_evaluated"]} seizures)')
    print(f'  Patient-level (weighted): {metrics["acc_pt_weighted"]:.4f}  '
          f'({metrics["n_patients_evaluated"]} patients)')
    print(f'  Patient-level (strict):   {metrics["acc_pt_strict"]:.4f}')
    print(f'  Patient-level (lenient):  {metrics["acc_pt_lenient"]:.4f}')

    if metrics.get('szunc_mean', 0) > 0:
        print(f'  Mean seizure uncertainty: {metrics["szunc_mean"]:.4f}')

    print(f'\n  Per-channel AUC:')
    for i, auc in enumerate(metrics['per_channel_auc']):
        a, b = TCP_PAIRS[i]
        print(f'    {a:3s}-{b:3s}: {auc:.4f}' if not np.isnan(auc) else f'    {a:3s}-{b:3s}: N/A')


def _save_results(results, output_dir):
    """Save metrics, per-seizure, and per-patient CSVs."""
    for mode_name, mode_results in results.items():
        prefix = mode_name  # 'single_pass' or 'mc_dropout'
        metrics = mode_results['metrics']

        # metrics JSON
        metrics_serializable = {
            k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
            for k, v in metrics.items()
        }
        with open(os.path.join(output_dir, f'soz_eval_{prefix}_metrics.json'), 'w') as f:
            json.dump(metrics_serializable, f, indent=2, default=str)

        # per-seizure CSV
        per_sz = mode_results['per_seizure']
        if per_sz:
            with open(os.path.join(output_dir, f'soz_eval_{prefix}_per_seizure.csv'), 'w',
                       newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    'patient_id', 'file_path', 'correct', 'pred_channel',
                    'pred_channel_idx', 'true_soz', 'unc_max',
                ])
                w.writeheader()
                for row in per_sz:
                    row_copy = dict(row)
                    row_copy['true_soz'] = ';'.join(row['true_soz'])
                    w.writerow(row_copy)

        # per-patient CSV
        per_pt = mode_results['per_patient']
        if per_pt:
            with open(os.path.join(output_dir, f'soz_eval_{prefix}_per_patient.csv'), 'w',
                       newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    'patient_id', 'correct_weighted', 'correct_strict',
                    'correct_lenient', 'weighted_score', 'n_seizures',
                    'n_patterns', 'soz_channels',
                ])
                w.writeheader()
                for row in per_pt:
                    row_copy = dict(row)
                    row_copy['soz_channels'] = ';'.join(row['soz_channels'])
                    w.writerow(row_copy)

    print(f'\n  Results saved to {output_dir}/')


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load model ──
    model = SimpleCNN(n_channels=22, task='soz').to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f'Loaded model from {args.model_path}')

    # ── Load test data ──
    test_ds = TUSZBaselineDataset(
        args.data_dir, split=args.split, task='soz', return_meta=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── Evaluate ──
    output_dir = args.output_dir or str(Path(args.model_path).parent)
    evaluate_soz(
        model, test_loader, device,
        output_dir=output_dir,
        mc_samples=args.mc_samples,
        neighbour_threshold=args.neighbour_threshold,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='DeepSOZ-style SOZ evaluation')
    p.add_argument('--model_path', type=str, required=True,
                   help='Path to trained SOZ model weights (.pt)')
    p.add_argument('--data_dir', type=str, default=r'F:\process_dataset\baseline')
    p.add_argument('--split', type=str, default='eval', choices=['dev', 'eval'])
    p.add_argument('--output_dir', type=str, default=None,
                   help='Output directory (default: same as model_path)')
    p.add_argument('--mc_samples', type=int, default=20,
                   help='MC dropout samples (0 = single pass only)')
    p.add_argument('--neighbour_threshold', type=int, default=4,
                   help='Max true SOZ channels for neighborhood relaxation')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    main(p.parse_args())
