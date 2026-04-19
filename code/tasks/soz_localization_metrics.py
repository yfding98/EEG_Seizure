#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSOZ-style SOZ localization evaluation metrics.

Two levels of evaluation aligned with the DeepSOZ paper:
  1. Seizure-level (corr_sz): MC dropout → normalize → mean → argmax → check SOZ
  2. Patient-level (acc_pt): Group by patient/SOZ-pattern → aggregate MC → final_loc

Supports both:
  - Lightweight mode (single forward pass, no MC) for per-epoch monitoring
  - Full MC dropout mode for end-of-training evaluation

References:
    Mandge & Bhatt, "DeepSOZ: A Robust Deep Model for Joint Temporal and
    Spatial Seizure Onset Zone Localization from Multichannel EEG Data",
    https://github.com/deeksha-ms/DeepSOZ
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    _HAS_TORCH = False


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel ordering
# ---------------------------------------------------------------------------

# Model output order (STANDARD_19 / BipolarToMonopolarMapper)
STANDARD_19 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ',
]

# DeepSOZ official 19-channel order
DEEPSOZ_19 = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3',  'C3',  'CZ', 'C4', 'T4',
    'T5',  'P3',  'PZ', 'P4', 'T6',
    'O1',  'O2',
]

# Build reorder index: STANDARD_19 → DeepSOZ
_std_idx = {ch: i for i, ch in enumerate(STANDARD_19)}
STANDARD_TO_DEEPSOZ = [_std_idx[ch] for ch in DEEPSOZ_19]


def reorder_to_deepsoz(arr: np.ndarray) -> np.ndarray:
    """Reorder [..., 19] array from STANDARD_19 to DeepSOZ order."""
    return arr[..., STANDARD_TO_DEEPSOZ]


# ---------------------------------------------------------------------------
# Spatial neighborhood (DeepSOZ official, indexed in DEEPSOZ_19 order)
# ---------------------------------------------------------------------------

CHN_NEIGHBOURS_19 = {
    0:  [1, 2, 3, 4],                    # FP1
    1:  [0, 4, 5, 6],                    # FP2
    2:  [0, 3, 4, 7, 8],                 # F7
    3:  [0, 2, 4, 8, 9],                 # F3
    4:  [0, 1, 3, 5, 9],                 # FZ
    5:  [1, 4, 6, 9, 10],                # F4
    6:  [1, 4, 5, 10, 11],               # F8
    7:  [2, 8, 12, 13, 17],              # T3
    8:  [2, 3, 4, 7, 9, 12, 13, 14],     # C3
    9:  [3, 4, 5, 8, 10, 13, 14, 15],    # CZ
    10: [4, 5, 6, 9, 11, 14, 15, 16],    # C4
    11: [6, 10, 15, 16, 18],             # T4
    12: [7, 8, 13, 17],                  # T5
    13: [7, 8, 9, 12, 14, 17],           # P3
    14: [8, 9, 10, 13, 15, 17, 18],      # PZ
    15: [9, 10, 11, 14, 16, 18],         # P4
    16: [10, 11, 15, 18],                # T6
    17: [7, 12, 13, 14, 18],             # O1
    18: [11, 14, 15, 16, 17],            # O2
}


def check_neighborhood(max_chn: int, onset_map: np.ndarray) -> bool:
    """Check if max_chn is a spatial neighbor of any true SOZ channel."""
    for i in range(len(onset_map)):
        if onset_map[i] == 1 and max_chn in CHN_NEIGHBOURS_19.get(i, []):
            return True
    return False


# ---------------------------------------------------------------------------
# Core: final_loc (from DeepSOZ official code)
# ---------------------------------------------------------------------------

def final_loc(
    psoz: np.ndarray,
    true_onset: np.ndarray,
    neighbour_threshold: int = 4,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    DeepSOZ official final_loc for SOZ localization evaluation.

    Parameters
    ----------
    psoz : [N, 19] SOZ probabilities from N MC samples (DeepSOZ channel order).
    true_onset : [19] binary ground truth SOZ map (DeepSOZ channel order).
    neighbour_threshold : Max number of true SOZ channels for neighborhood
        relaxation to apply.

    Returns
    -------
    ysoz : [19] averaged normalized SOZ scores.
    uncertainty : [19] variance across MC samples.
    correct : 1 if localization is correct, 0 otherwise.
    """
    if psoz.ndim == 1:
        psoz = psoz[np.newaxis, :]

    m = psoz.max(axis=1, keepdims=True)
    m = np.where(m > 1e-12, m, 1.0)
    psoz_norm = psoz / m
    ysoz = psoz_norm.mean(axis=0)

    max_chn = int(np.argmax(ysoz))
    correct = 1 if true_onset[max_chn] == 1 else 0

    if (correct == 0
            and int(true_onset.sum()) <= neighbour_threshold
            and check_neighborhood(max_chn, true_onset)):
        correct = 1

    uncertainty = psoz_norm.var(axis=0)
    uncertainty = np.nan_to_num(uncertainty, nan=0.0)
    return ysoz, uncertainty, correct


def _onset_key(onset_map: np.ndarray) -> str:
    """Hash an onset map for grouping by SOZ pattern."""
    return ','.join(str(int(v)) for v in onset_map)


# ---------------------------------------------------------------------------
# Lightweight: single-pass seizure & patient-level metrics
# ---------------------------------------------------------------------------

def compute_deepsoz_soz_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    patient_ids: List[str],
    edf_paths: List[str],
    neighbour_threshold: int = 4,
) -> Dict[str, float]:
    """
    Lightweight DeepSOZ SOZ localization metrics (single forward pass).

    Uses each sample's probability vector as a single "MC sample" to compute
    seizure-level and patient-level accuracy with neighborhood relaxation.
    No uncertainty estimation (requires MC dropout).

    Parameters
    ----------
    probs : [N, 19] SOZ probabilities (STANDARD_19 order).
    targets : [N, 19] binary SOZ labels (STANDARD_19 order).
    patient_ids : List of patient IDs for each sample.
    edf_paths : List of EDF paths for each sample.
    neighbour_threshold : Neighborhood relaxation threshold.

    Returns
    -------
    Flat dict with corr_sz, acc_pt_weighted, acc_pt_strict, acc_pt_lenient.
    """
    n = probs.shape[0]
    if n == 0:
        return {
            'corr_sz': 0.0,
            'acc_pt_weighted': 0.0,
            'acc_pt_strict': 0.0,
            'acc_pt_lenient': 0.0,
            'n_seizures': 0,
            'n_patients': 0,
        }

    # Reorder to DeepSOZ channel order
    probs_dsz = reorder_to_deepsoz(probs)
    targets_dsz = reorder_to_deepsoz(targets)

    # --- Seizure-level ---
    # Each sample treated as single MC sample → final_loc
    seizure_correct = []
    # Group by patient for patient-level
    patient_seizures: Dict[str, List[Dict]] = defaultdict(list)

    for i in range(n):
        true_onset = targets_dsz[i]
        if true_onset.sum() == 0:
            continue  # Skip samples with no SOZ annotation

        psoz_single = probs_dsz[i:i+1, :]  # [1, 19]
        _, _, correct = final_loc(psoz_single, true_onset, neighbour_threshold)
        seizure_correct.append(correct)

        pid = patient_ids[i]
        onset_key = _onset_key(true_onset)
        patient_seizures[pid].append({
            'onset_key': onset_key,
            'onset_map': true_onset.copy(),
            'probs': probs_dsz[i],
            'correct': correct,
        })

    corr_sz = float(np.mean(seizure_correct)) if seizure_correct else 0.0

    # --- Patient-level ---
    n_patients = len(patient_seizures)
    pt_weighted_correct = 0
    pt_strict_correct = 0
    pt_lenient_correct = 0

    for pid, seizures in patient_seizures.items():
        # Group by SOZ pattern within patient
        pattern_groups: Dict[str, Dict] = {}
        for sz in seizures:
            key = sz['onset_key']
            if key not in pattern_groups:
                pattern_groups[key] = {
                    'onset_map': sz['onset_map'],
                    'probs_list': [],
                    'n_seizures': 0,
                }
            pattern_groups[key]['probs_list'].append(sz['probs'])
            pattern_groups[key]['n_seizures'] += 1

        # Evaluate each pattern group
        pattern_results = []
        for key, group in pattern_groups.items():
            group_psoz = np.stack(group['probs_list'], axis=0)  # [M, 19]
            _, _, correct_g = final_loc(
                group_psoz, group['onset_map'], neighbour_threshold
            )
            pattern_results.append({
                'correct': correct_g,
                'n_seizures': group['n_seizures'],
            })

        total_sz_in_pt = sum(g['n_seizures'] for g in pattern_results)
        weighted_correct = sum(
            g['correct'] * g['n_seizures'] / max(total_sz_in_pt, 1)
            for g in pattern_results
        )
        all_correct = int(all(g['correct'] for g in pattern_results))
        any_correct = int(any(g['correct'] for g in pattern_results))

        pt_weighted_correct += int(weighted_correct >= 0.5)
        pt_strict_correct += all_correct
        pt_lenient_correct += any_correct

    return {
        'corr_sz': corr_sz,
        'acc_pt_weighted': pt_weighted_correct / max(n_patients, 1),
        'acc_pt_strict': pt_strict_correct / max(n_patients, 1),
        'acc_pt_lenient': pt_lenient_correct / max(n_patients, 1),
        'n_seizures': len(seizure_correct),
        'n_patients': n_patients,
    }


# ---------------------------------------------------------------------------
# Full MC dropout: seizure & patient-level with uncertainty
# ---------------------------------------------------------------------------

def mc_inference(
    model,
    loader,
    device,
    mc_samples: int = 20,
) -> Dict[str, Any]:
    """
    Run MC dropout inference on all samples in loader.

    Parameters
    ----------
    model : The model (will be set to train mode for dropout).
    loader : DataLoader yielding batches with keys:
        x, onset_sec, start_sec, label, patient_id, edf_path,
        and optionally brain_nets, valid_patch_counts, rel_time.
    device : torch device.
    mc_samples : Number of MC forward passes.

    Returns
    -------
    Dict with:
        mc_probs: List of [mc_samples, 19] arrays per sample (STANDARD_19 order).
        targets: [N, 19] array (STANDARD_19 order).
        patient_ids: List[str], edf_paths: List[str].
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required for MC inference")

    all_mc_probs = []   # List of [mc_samples, 19] per sample
    all_targets = []
    all_patient_ids = []
    all_edf_paths = []

    for batch in loader:
        x = batch['x'].to(device)
        onset = batch['onset_sec'].to(device)
        start = batch['start_sec'].to(device)
        labels = batch['label'].cpu().numpy()
        pids = batch['patient_id']
        edfs = batch['edf_path']

        brain_nets = batch.get('brain_nets', None)
        vp_counts = batch.get('valid_patch_counts', None)
        rel_time = batch.get('rel_time', None)
        if brain_nets is not None:
            brain_nets = brain_nets.to(device)
        if vp_counts is not None:
            vp_counts = vp_counts.to(device)
        if rel_time is not None:
            rel_time = rel_time.to(device)

        B = x.size(0)

        # MC dropout: model.train() keeps dropout active
        model.train()
        mc_batch = []  # [mc_samples, B, 19]

        for _ in range(mc_samples):
            with torch.no_grad():
                out = model(
                    x, onset, start,
                    valid_patch_counts=vp_counts,
                    brain_networks=brain_nets,
                    rel_time=rel_time,
                )
            mc_batch.append(out['soz_probs'].cpu().numpy())  # [B, 19]

        model.eval()
        mc_batch = np.stack(mc_batch, axis=0)  # [mc_samples, B, 19]

        for i in range(B):
            all_mc_probs.append(mc_batch[:, i, :])  # [mc_samples, 19]
            all_targets.append(labels[i])
            all_patient_ids.append(pids[i])
            all_edf_paths.append(edfs[i])

    return {
        'mc_probs': all_mc_probs,
        'targets': np.stack(all_targets, axis=0),
        'patient_ids': all_patient_ids,
        'edf_paths': all_edf_paths,
    }


def compute_deepsoz_soz_metrics_mc(
    model,
    loader,
    device,
    mc_samples: int = 20,
    neighbour_threshold: int = 4,
) -> Dict[str, Any]:
    """
    Full DeepSOZ SOZ localization evaluation with MC dropout.

    Returns seizure-level and patient-level metrics with uncertainty.
    """
    # Run MC inference
    mc_data = mc_inference(model, loader, device, mc_samples=mc_samples)
    mc_probs_list = mc_data['mc_probs']       # List of [mc_samples, 19] (STANDARD_19)
    targets = mc_data['targets']               # [N, 19] (STANDARD_19)
    patient_ids = mc_data['patient_ids']
    edf_paths = mc_data['edf_paths']
    n = len(mc_probs_list)

    if n == 0:
        return {
            'metrics': {
                'corr_sz': 0.0, 'szunc_mean': 0.0,
                'acc_pt_weighted': 0.0, 'acc_pt_strict': 0.0,
                'acc_pt_lenient': 0.0, 'ptunc_mean': 0.0,
                'n_seizures': 0, 'n_patients': 0,
            },
            'per_seizure': [],
            'per_patient': [],
        }

    # --- Seizure-level with MC ---
    seizure_correct = []
    seizure_unc = []
    per_seizure = []

    # Also collect for patient-level grouping
    patient_data: Dict[str, List[Dict]] = defaultdict(list)

    for i in range(n):
        mc_probs_dsz = reorder_to_deepsoz(mc_probs_list[i])  # [mc_samples, 19]
        true_onset_dsz = reorder_to_deepsoz(targets[i])        # [19]

        if true_onset_dsz.sum() == 0:
            continue

        ysoz, unc, correct = final_loc(
            mc_probs_dsz, true_onset_dsz, neighbour_threshold
        )
        seizure_correct.append(correct)
        seizure_unc.append(unc)

        soz_channels = [DEEPSOZ_19[j] for j in range(19) if true_onset_dsz[j] == 1]
        per_seizure.append({
            'patient_id': patient_ids[i],
            'edf_path': edf_paths[i],
            'correct': correct,
            'max_chn': int(np.argmax(ysoz)),
            'max_chn_name': DEEPSOZ_19[int(np.argmax(ysoz))],
            'true_soz': soz_channels,
            'unc_max': float(unc.max()),
        })

        pid = patient_ids[i]
        onset_key = _onset_key(true_onset_dsz)
        patient_data[pid].append({
            'onset_key': onset_key,
            'onset_map': true_onset_dsz.copy(),
            'mc_probs_dsz': mc_probs_dsz,
            'correct': correct,
        })

    corr_sz = float(np.mean(seizure_correct)) if seizure_correct else 0.0
    szunc_mean = float(np.mean([u.max() for u in seizure_unc])) if seizure_unc else 0.0

    # --- Patient-level with MC ---
    n_patients = len(patient_data)
    pt_weighted_correct = 0
    pt_strict_correct = 0
    pt_lenient_correct = 0
    pt_uncs = []
    per_patient = []

    for pid, seizures in patient_data.items():
        # Group by SOZ pattern
        pattern_groups: Dict[str, Dict] = {}
        for sz in seizures:
            key = sz['onset_key']
            if key not in pattern_groups:
                pattern_groups[key] = {
                    'onset_map': sz['onset_map'],
                    'mc_probs_list': [],
                    'n_seizures': 0,
                }
            pattern_groups[key]['mc_probs_list'].append(sz['mc_probs_dsz'])
            pattern_groups[key]['n_seizures'] += 1

        pattern_results = []
        pattern_uncs = []
        for key, group in pattern_groups.items():
            # Aggregate all MC samples from all seizures in this pattern
            group_psoz = np.concatenate(group['mc_probs_list'], axis=0)
            ysoz_g, unc_g, correct_g = final_loc(
                group_psoz, group['onset_map'], neighbour_threshold
            )
            soz_channels = [DEEPSOZ_19[j] for j in range(19) if group['onset_map'][j] == 1]
            pattern_results.append({
                'correct': correct_g,
                'n_seizures': group['n_seizures'],
                'soz_channels': soz_channels,
                'max_chn_name': DEEPSOZ_19[int(np.argmax(ysoz_g))],
                'unc_max': float(unc_g.max()),
            })
            pattern_uncs.append(unc_g)

        total_sz = sum(g['n_seizures'] for g in pattern_results)
        weighted_correct = sum(
            g['correct'] * g['n_seizures'] / max(total_sz, 1)
            for g in pattern_results
        )
        all_correct = int(all(g['correct'] for g in pattern_results))
        any_correct = int(any(g['correct'] for g in pattern_results))

        pt_correct = int(weighted_correct >= 0.5)
        pt_weighted_correct += pt_correct
        pt_strict_correct += all_correct
        pt_lenient_correct += any_correct

        pt_unc = float(np.mean([u.max() for u in pattern_uncs])) if pattern_uncs else 0.0
        pt_uncs.append(pt_unc)

        all_soz = set()
        for g in pattern_results:
            all_soz.update(g.get('soz_channels', []))

        per_patient.append({
            'patient_id': pid,
            'correct_weighted': pt_correct,
            'correct_all': all_correct,
            'correct_any': any_correct,
            'weighted_score': round(weighted_correct, 3),
            'n_seizures': total_sz,
            'n_patterns': len(pattern_groups),
            'patterns': pattern_results,
            'all_soz': sorted(all_soz),
            'unc_max': pt_unc,
        })

    ptunc_mean = float(np.mean(pt_uncs)) if pt_uncs else 0.0

    metrics = {
        'corr_sz': corr_sz,
        'szunc_mean': szunc_mean,
        'acc_pt_weighted': pt_weighted_correct / max(n_patients, 1),
        'acc_pt_strict': pt_strict_correct / max(n_patients, 1),
        'acc_pt_lenient': pt_lenient_correct / max(n_patients, 1),
        'ptunc_mean': ptunc_mean,
        'n_seizures': len(seizure_correct),
        'n_patients': n_patients,
    }

    return {
        'metrics': metrics,
        'per_seizure': per_seizure,
        'per_patient': per_patient,
    }


# ---------------------------------------------------------------------------
# Detailed evaluation with file output
# ---------------------------------------------------------------------------

def run_detailed_soz_evaluation(
    model,
    loader,
    device,
    mc_samples: int = 20,
    neighbour_threshold: int = 4,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full MC-based SOZ localization evaluation and optionally save results.
    """
    results = compute_deepsoz_soz_metrics_mc(
        model, loader, device,
        mc_samples=mc_samples,
        neighbour_threshold=neighbour_threshold,
    )
    metrics = results['metrics']

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON summary
        metrics_path = out / 'soz_deepsoz_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        log.info("Saved SOZ DeepSOZ metrics -> %s", metrics_path)

        # Per-seizure CSV
        if results['per_seizure']:
            csv_path = out / 'soz_deepsoz_per_seizure.csv'
            try:
                import pandas as pd
                rows = []
                for s in results['per_seizure']:
                    rows.append({
                        'patient_id': s['patient_id'],
                        'edf_path': s['edf_path'],
                        'correct': s['correct'],
                        'max_chn': s['max_chn'],
                        'max_chn_name': s['max_chn_name'],
                        'true_soz': '|'.join(s['true_soz']),
                        'unc_max': s['unc_max'],
                    })
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                log.info("Saved per-seizure SOZ results -> %s", csv_path)
            except ImportError:
                pass

        # Per-patient CSV
        if results['per_patient']:
            csv_path = out / 'soz_deepsoz_per_patient.csv'
            try:
                import pandas as pd
                rows = []
                for p in results['per_patient']:
                    rows.append({
                        'patient_id': p['patient_id'],
                        'correct_weighted': p['correct_weighted'],
                        'correct_all': p['correct_all'],
                        'correct_any': p['correct_any'],
                        'weighted_score': p['weighted_score'],
                        'n_seizures': p['n_seizures'],
                        'n_patterns': p['n_patterns'],
                        'all_soz': '|'.join(p['all_soz']),
                        'unc_max': p['unc_max'],
                    })
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                log.info("Saved per-patient SOZ results -> %s", csv_path)
            except ImportError:
                pass

    return results
