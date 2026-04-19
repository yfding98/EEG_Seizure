#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSOZ-style seizure detection evaluation metrics.

Two levels of evaluation aligned with the DeepSOZ paper:
  1. Window-level:  AU-ROC, Sensitivity, Specificity  (per-patch binary)
  2. Seizure-level: FPR/hr, Sensitivity, Latency      (per-event, with smoothing)

All functions are pure (numpy/sklearn/torch.nn.functional only) and have no
dependency on the training script or model classes.

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
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc
except ImportError:
    _sklearn_roc_auc = None

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    _HAS_TORCH = False


log = logging.getLogger(__name__)

# Type alias for a seizure-event key
SeizureKey = Tuple[str, str, float, float]  # (patient_id, edf_path, sz_start, sz_end)


# ---------------------------------------------------------------------------
# 1. Window-level metrics
# ---------------------------------------------------------------------------

def compute_window_level_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute DeepSOZ window-level metrics: AU-ROC, Sensitivity, Specificity.

    Parameters
    ----------
    probs  : 1-D array of seizure-class probabilities (softmax output).
    labels : 1-D array of ground-truth binary labels (0 or 1).
    threshold : decision threshold for Sensitivity/Specificity.

    Returns
    -------
    Dict with keys: window_auroc, window_sensitivity, window_specificity.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if probs.size == 0:
        return {'window_auroc': 0.0, 'window_sensitivity': 0.0, 'window_specificity': 0.0}

    # AU-ROC (threshold-free)
    auroc = 0.0
    if _sklearn_roc_auc is not None and np.unique(labels).size > 1:
        try:
            auroc = float(_sklearn_roc_auc(labels, probs))
        except ValueError:
            auroc = 0.0

    # Hard-decision metrics
    preds = (probs >= threshold).astype(np.int64)
    tp = float(np.logical_and(preds == 1, labels == 1).sum())
    fp = float(np.logical_and(preds == 1, labels == 0).sum())
    tn = float(np.logical_and(preds == 0, labels == 0).sum())
    fn = float(np.logical_and(preds == 0, labels == 1).sum())

    sensitivity = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)

    return {
        'window_auroc': auroc,
        'window_sensitivity': sensitivity,
        'window_specificity': specificity,
    }


# ---------------------------------------------------------------------------
# 2. Grouping & merging
# ---------------------------------------------------------------------------

def group_patches_by_seizure(
    records: Dict[str, List],
) -> Dict[SeizureKey, Dict[str, np.ndarray]]:
    """
    Group flat patch records by seizure event.

    Parameters
    ----------
    records : dict of parallel lists with keys:
        patient_id, edf_path, seizure_start_sec, seizure_end_sec,
        patch_abs_start_sec, prob_seizure, label.

    Returns
    -------
    Dict mapping SeizureKey -> {patch_abs_start_sec, prob_seizure, label}
    as sorted numpy arrays.
    """
    groups: Dict[SeizureKey, Dict[str, list]] = defaultdict(
        lambda: {'patch_abs_start_sec': [], 'prob_seizure': [], 'label': []}
    )
    n = len(records['patient_id'])
    for i in range(n):
        key: SeizureKey = (
            records['patient_id'][i],
            records['edf_path'][i],
            float(records['seizure_start_sec'][i]),
            float(records['seizure_end_sec'][i]),
        )
        groups[key]['patch_abs_start_sec'].append(records['patch_abs_start_sec'][i])
        groups[key]['prob_seizure'].append(records['prob_seizure'][i])
        groups[key]['label'].append(records['label'][i])

    result: Dict[SeizureKey, Dict[str, np.ndarray]] = {}
    for key, data in groups.items():
        order = np.argsort(data['patch_abs_start_sec'])
        result[key] = {
            'patch_abs_start_sec': np.array(data['patch_abs_start_sec'], dtype=np.float64)[order],
            'prob_seizure': np.array(data['prob_seizure'], dtype=np.float64)[order],
            'label': np.array(data['label'], dtype=np.int64)[order],
        }
    return result


def merge_overlapping_patches(
    group: Dict[str, np.ndarray],
    patch_duration_sec: float,
) -> Dict[str, np.ndarray]:
    """
    Merge temporally overlapping patches from multiple windows (onset/mid/offset).

    When multiple windows cover the same seizure event, patches at the same
    temporal position are averaged (probabilities) or max-ed (labels).

    Returns dict with keys: time_grid, prob_seizure, label.
    """
    starts = group['patch_abs_start_sec']
    probs = group['prob_seizure']
    labels = group['label']

    if len(starts) == 0:
        return {
            'time_grid': np.array([], dtype=np.float64),
            'prob_seizure': np.array([], dtype=np.float64),
            'label': np.array([], dtype=np.int64),
        }

    tol = patch_duration_sec * 0.5
    t_min = starts[0]
    t_max = starts[-1]

    # Build regular grid
    n_grid = max(1, int(np.round((t_max - t_min) / patch_duration_sec)) + 1)
    time_grid = t_min + np.arange(n_grid) * patch_duration_sec

    merged_probs = np.zeros(n_grid, dtype=np.float64)
    merged_labels = np.zeros(n_grid, dtype=np.int64)
    counts = np.zeros(n_grid, dtype=np.float64)

    for i, t in enumerate(starts):
        idx = int(np.round((t - t_min) / patch_duration_sec))
        idx = min(idx, n_grid - 1)
        if abs(time_grid[idx] - t) <= tol:
            merged_probs[idx] += probs[i]
            merged_labels[idx] = max(merged_labels[idx], labels[i])
            counts[idx] += 1.0

    # Average probabilities where multiple patches overlap
    valid = counts > 0
    merged_probs[valid] /= counts[valid]

    return {
        'time_grid': time_grid,
        'prob_seizure': merged_probs,
        'label': merged_labels,
    }


# ---------------------------------------------------------------------------
# 3. Moving average smoother
# ---------------------------------------------------------------------------

def apply_moving_average(
    probs: np.ndarray,
    kernel_size: int = 31,
) -> np.ndarray:
    """
    Apply moving average smoothing identical to DeepSOZ's
    ``nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)``.

    Uses ``torch.nn.functional.avg_pool1d`` when available for exact parity.
    Falls back to numpy convolution otherwise.
    """
    n = len(probs)
    if n == 0:
        return probs.copy()

    # Adapt kernel for short sequences
    k = min(kernel_size, n)
    if k % 2 == 0:
        k = max(k - 1, 1)

    if _HAS_TORCH:
        pad = k // 2
        t = torch.tensor(probs, dtype=torch.float64).view(1, 1, -1)
        smoothed = F.avg_pool1d(
            F.pad(t, (pad, pad), mode='constant', value=0.0),
            kernel_size=k,
            stride=1,
        )
        return smoothed.view(-1).numpy()
    else:
        # Numpy fallback: zero-pad then convolve with mode='valid'
        pad = k // 2
        padded = np.pad(probs, (pad, pad), mode='constant', constant_values=0.0)
        kernel = np.ones(k, dtype=np.float64) / k
        return np.convolve(padded, kernel, mode='valid')[:n]


# ---------------------------------------------------------------------------
# 4. Threshold search
# ---------------------------------------------------------------------------

def search_optimal_threshold(
    seizure_groups: Dict[SeizureKey, Dict[str, np.ndarray]],
    patch_duration_sec: float,
    threshold_range: Tuple[float, float] = (0.3, 0.75),
    threshold_step: float = 0.01,
    max_fpr_per_hour: float = 120.0,
) -> Tuple[float, float]:
    """
    Search for the lowest threshold where FPR/hr <= max_fpr_per_hour.

    This mirrors DeepSOZ's threshold selection on the validation set.

    Parameters
    ----------
    seizure_groups : Already smoothed and merged groups keyed by SeizureKey.
        Each value dict must have 'prob_seizure', 'label', 'time_grid'.
    patch_duration_sec : Duration of one patch in seconds.
    threshold_range : (low, high) threshold sweep range.
    threshold_step : Step size for the sweep.
    max_fpr_per_hour : FPR constraint (DeepSOZ uses 120).

    Returns
    -------
    (optimal_threshold, fpr_at_threshold)
    """
    thresholds = np.arange(
        threshold_range[0],
        threshold_range[1] + threshold_step * 0.5,
        threshold_step,
    )

    best_threshold = threshold_range[1]
    best_fpr = float('inf')

    for thres in thresholds:
        total_fp = 0.0
        total_preictal_hours = 0.0

        for key, group in seizure_groups.items():
            sz_start = key[2]
            probs = group['prob_seizure']
            labels = group['label']
            times = group['time_grid']

            # Pre-ictal: patches before seizure onset with label == 0
            preictal_mask = (times < sz_start) & (labels == 0)
            n_preictal = int(preictal_mask.sum())
            if n_preictal > 0:
                total_preictal_hours += n_preictal * patch_duration_sec / 3600.0
                total_fp += float((probs[preictal_mask] >= thres).sum())

        if total_preictal_hours > 0:
            fpr = total_fp / total_preictal_hours
        else:
            fpr = 0.0

        if fpr <= max_fpr_per_hour:
            return float(thres), float(fpr)

        if fpr < best_fpr:
            best_fpr = fpr
            best_threshold = float(thres)

    return best_threshold, best_fpr


# ---------------------------------------------------------------------------
# 5. Seizure-level metrics
# ---------------------------------------------------------------------------

def compute_seizure_level_metrics(
    seizure_groups: Dict[SeizureKey, Dict[str, np.ndarray]],
    threshold: float,
    patch_duration_sec: float,
) -> Dict[str, float]:
    """
    Compute seizure-level metrics: Sensitivity, FPR/hr, Latency.

    Parameters
    ----------
    seizure_groups : Smoothed & merged groups keyed by SeizureKey.
    threshold : Decision threshold on smoothed probabilities.
    patch_duration_sec : Duration of one patch in seconds.

    Returns
    -------
    Dict with seizure_sensitivity, fpr_per_hour, mean_latency_sec,
    median_latency_sec, n_seizures_detected, n_seizures_total.
    """
    n_total = len(seizure_groups)
    n_detected = 0
    latencies: List[float] = []
    total_fp = 0.0
    total_preictal_hours = 0.0

    for key, group in seizure_groups.items():
        sz_start = key[2]
        sz_end = key[3]
        probs = group['prob_seizure']
        labels = group['label']
        times = group['time_grid']

        if len(probs) == 0:
            continue

        # Binary predictions after smoothing
        preds = (probs >= threshold).astype(np.int64)

        # --- Seizure-level sensitivity ---
        # A seizure is "detected" if ANY patch within [sz_start, sz_end) is predicted positive
        ictal_mask = (times >= sz_start) & (times < sz_end)
        if ictal_mask.any() and preds[ictal_mask].any():
            n_detected += 1

        # --- FPR per hour ---
        preictal_mask = (times < sz_start) & (labels == 0)
        n_preictal = int(preictal_mask.sum())
        if n_preictal > 0:
            total_preictal_hours += n_preictal * patch_duration_sec / 3600.0
            total_fp += float(preds[preictal_mask].sum())

        # --- Latency ---
        # Find rising edges in prediction sequence
        diff = np.diff(np.concatenate([[0], preds]))
        onsets = np.where(diff == 1)[0]  # indices where prediction transitions 0->1

        if len(onsets) > 0:
            # Find first onset that overlaps with the true seizure period
            for onset_idx in onsets:
                # Find corresponding offset (next falling edge)
                offsets_after = np.where(
                    (np.diff(np.concatenate([preds, [0]])) == -1)
                    & (np.arange(len(preds)) >= onset_idx)
                )[0]
                if len(offsets_after) > 0:
                    offset_idx = offsets_after[0] + 1  # exclusive end
                else:
                    offset_idx = len(preds)

                # Check if this predicted segment overlaps with true seizure
                pred_start = times[onset_idx]
                pred_end = times[min(offset_idx, len(times) - 1)]
                if pred_end >= sz_start and pred_start < sz_end:
                    latency = pred_start - sz_start
                    latencies.append(latency)
                    break

    # Aggregate
    seizure_sensitivity = n_detected / max(n_total, 1)
    fpr_per_hour = total_fp / max(total_preictal_hours, 1e-12)
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    median_latency = float(np.median(latencies)) if latencies else 0.0

    return {
        'seizure_sensitivity': seizure_sensitivity,
        'fpr_per_hour': fpr_per_hour,
        'mean_latency_sec': mean_latency,
        'median_latency_sec': median_latency,
        'n_seizures_detected': n_detected,
        'n_seizures_total': n_total,
        'n_latency_samples': len(latencies),
    }


# ---------------------------------------------------------------------------
# 6. Main orchestrator
# ---------------------------------------------------------------------------

def compute_deepsoz_stage_metrics(
    records: Dict[str, List],
    patch_duration_sec: float,
    smoother_kernel_size: int = 31,
    threshold: Optional[float] = None,
    threshold_range: Tuple[float, float] = (0.3, 0.75),
    threshold_step: float = 0.01,
    max_fpr_per_hour: float = 120.0,
) -> Dict[str, float]:
    """
    Compute all DeepSOZ-style metrics (window-level + seizure-level).

    Parameters
    ----------
    records : dict of parallel lists from evaluate_stage(collect_temporal=True).
    patch_duration_sec : Duration of one patch in seconds.
    smoother_kernel_size : Moving average kernel size (odd int, default 31).
    threshold : If provided, skip threshold search and use this value.
    threshold_range : (low, high) for threshold sweep.
    threshold_step : Step size for threshold sweep.
    max_fpr_per_hour : FPR constraint for threshold selection.

    Returns
    -------
    Flat dict with all metrics (suitable for TensorBoard logging).
    """
    n_records = len(records.get('patient_id', []))
    if n_records == 0:
        return {
            'window_auroc': 0.0,
            'window_sensitivity': 0.0,
            'window_specificity': 0.0,
            'optimal_threshold': 0.5,
            'seizure_sensitivity': 0.0,
            'fpr_per_hour': 0.0,
            'mean_latency_sec': 0.0,
            'median_latency_sec': 0.0,
            'n_seizures_detected': 0,
            'n_seizures_total': 0,
            'n_latency_samples': 0,
        }

    all_probs = np.array(records['prob_seizure'], dtype=np.float64)
    all_labels = np.array(records['label'], dtype=np.int64)

    # --- Window-level metrics (on raw probs, no smoothing) ---
    window_metrics = compute_window_level_metrics(all_probs, all_labels)

    # --- Group by seizure event ---
    raw_groups = group_patches_by_seizure(records)

    # --- Merge overlapping patches & apply smoothing ---
    smoothed_groups: Dict[SeizureKey, Dict[str, np.ndarray]] = {}
    for key, group in raw_groups.items():
        merged = merge_overlapping_patches(group, patch_duration_sec)
        smoothed_probs = apply_moving_average(
            merged['prob_seizure'], kernel_size=smoother_kernel_size
        )
        smoothed_groups[key] = {
            'time_grid': merged['time_grid'],
            'prob_seizure': smoothed_probs,
            'label': merged['label'],
        }

    # --- Threshold search (or use provided) ---
    if threshold is None:
        optimal_threshold, fpr_at_threshold = search_optimal_threshold(
            smoothed_groups,
            patch_duration_sec,
            threshold_range=threshold_range,
            threshold_step=threshold_step,
            max_fpr_per_hour=max_fpr_per_hour,
        )
    else:
        optimal_threshold = float(threshold)
        fpr_at_threshold = 0.0

    # --- Seizure-level metrics ---
    seizure_metrics = compute_seizure_level_metrics(
        smoothed_groups, optimal_threshold, patch_duration_sec
    )

    # Combine
    result = {**window_metrics, 'optimal_threshold': optimal_threshold, **seizure_metrics}
    return result


# ---------------------------------------------------------------------------
# 7. Detailed evaluation (standalone use)
# ---------------------------------------------------------------------------

def run_detailed_evaluation(
    records: Dict[str, List],
    patch_duration_sec: float,
    smoother_kernel_size: int = 31,
    threshold: Optional[float] = None,
    threshold_range: Tuple[float, float] = (0.3, 0.75),
    threshold_step: float = 0.01,
    max_fpr_per_hour: float = 120.0,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute DeepSOZ metrics with per-seizure breakdown.

    Returns all metrics plus a per_seizure list with per-event details.
    Optionally saves results to output_dir.
    """
    metrics = compute_deepsoz_stage_metrics(
        records=records,
        patch_duration_sec=patch_duration_sec,
        smoother_kernel_size=smoother_kernel_size,
        threshold=threshold,
        threshold_range=threshold_range,
        threshold_step=threshold_step,
        max_fpr_per_hour=max_fpr_per_hour,
    )
    optimal_threshold = metrics['optimal_threshold']

    # Per-seizure breakdown
    raw_groups = group_patches_by_seizure(records)
    per_seizure: List[Dict[str, Any]] = []

    for key, group in raw_groups.items():
        patient_id, edf_path, sz_start, sz_end = key
        merged = merge_overlapping_patches(group, patch_duration_sec)
        smoothed_probs = apply_moving_average(
            merged['prob_seizure'], kernel_size=smoother_kernel_size
        )
        times = merged['time_grid']
        labels = merged['label']
        preds = (smoothed_probs >= optimal_threshold).astype(np.int64)

        # Ictal region
        ictal_mask = (times >= sz_start) & (times < sz_end)
        detected = bool(ictal_mask.any() and preds[ictal_mask].any())

        # Pre-ictal FP
        preictal_mask = (times < sz_start) & (labels == 0)
        n_preictal = int(preictal_mask.sum())
        n_preictal_fp = int(preds[preictal_mask].sum()) if n_preictal > 0 else 0

        # Latency
        latency = None
        diff = np.diff(np.concatenate([[0], preds]))
        onsets = np.where(diff == 1)[0]
        for onset_idx in onsets:
            offsets_after = np.where(
                (np.diff(np.concatenate([preds, [0]])) == -1)
                & (np.arange(len(preds)) >= onset_idx)
            )[0]
            offset_idx = (offsets_after[0] + 1) if len(offsets_after) > 0 else len(preds)
            pred_start = times[onset_idx]
            pred_end = times[min(offset_idx, len(times) - 1)]
            if pred_end >= sz_start and pred_start < sz_end:
                latency = float(pred_start - sz_start)
                break

        per_seizure.append({
            'patient_id': patient_id,
            'edf_path': edf_path,
            'seizure_start_sec': sz_start,
            'seizure_end_sec': sz_end,
            'seizure_duration_sec': sz_end - sz_start,
            'n_patches': len(times),
            'n_ictal_patches': int(ictal_mask.sum()),
            'n_preictal_patches': n_preictal,
            'n_preictal_fp': n_preictal_fp,
            'detected': detected,
            'latency_sec': latency,
        })

    result = {
        'metrics': metrics,
        'per_seizure': per_seizure,
    }

    # Save if output_dir provided
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON summary
        metrics_path = out / 'stage_deepsoz_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        log.info("Saved DeepSOZ metrics -> %s", metrics_path)

        # CSV per-seizure
        csv_path = out / 'stage_deepsoz_per_seizure.csv'
        try:
            import pandas as pd
            df = pd.DataFrame(per_seizure)
            df.to_csv(csv_path, index=False)
            log.info("Saved per-seizure breakdown -> %s", csv_path)
        except ImportError:
            # Fallback: write CSV manually
            import csv
            if per_seizure:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=per_seizure[0].keys())
                    writer.writeheader()
                    writer.writerows(per_seizure)
                log.info("Saved per-seizure breakdown -> %s", csv_path)

    return result
