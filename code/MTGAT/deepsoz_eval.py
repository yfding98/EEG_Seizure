#!/usr/bin/env python3
"""DeepSOZ-style SOZ localization metrics for MTGAT outputs.

This mirrors the evaluation idea in code/baseline/evaluate.py, but it uses the
MTGAT/private-manifest TCP channel order:
left temporal, right temporal, left parasagittal, right parasagittal, central.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None

try:
    from .soz_manifest import TCP_NAMES, TCP_PAIRS
except ImportError:  # Allows direct script imports from code/MTGAT
    from soz_manifest import TCP_NAMES, TCP_PAIRS


def _build_bipolar_neighbors() -> Dict[int, set[int]]:
    """Build neighbors by chain adjacency or shared monopolar electrode."""
    neighbors: Dict[int, set[int]] = defaultdict(set)
    chains = [
        [0, 1, 2, 3],        # left temporal
        [4, 5, 6, 7],        # right temporal
        [8, 9, 10, 11],      # left parasagittal
        [12, 13, 14, 15],    # right parasagittal
        [16, 17, 18, 19, 20, 21],  # central
    ]
    for chain in chains:
        for i in range(len(chain) - 1):
            neighbors[chain[i]].add(chain[i + 1])
            neighbors[chain[i + 1]].add(chain[i])

    for i, pair_i in enumerate(TCP_PAIRS):
        elecs_i = set(pair_i)
        for j in range(i + 1, len(TCP_PAIRS)):
            if elecs_i & set(TCP_PAIRS[j]):
                neighbors[i].add(j)
                neighbors[j].add(i)
    return dict(neighbors)


BIPOLAR_NEIGHBORS = _build_bipolar_neighbors()


def check_neighborhood(max_chn: int, onset_map: np.ndarray) -> bool:
    """Check whether the predicted channel neighbors any true SOZ channel."""
    for i, active in enumerate(onset_map):
        if active == 1 and max_chn in BIPOLAR_NEIGHBORS.get(i, set()):
            return True
    return False


def final_loc(
    psoz: np.ndarray,
    true_onset: np.ndarray,
    neighbour_threshold: int = 4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """DeepSOZ-style final localization on one seizure/patient group.

    psoz is [N, 22], where N can be windows, seizures, or MC samples.
    The rows are max-normalized, averaged, and evaluated by argmax with
    optional neighborhood relaxation.
    """
    if psoz.ndim == 1:
        psoz = psoz[np.newaxis, :]

    row_max = np.clip(psoz.max(axis=1, keepdims=True), 1e-8, None)
    psoz_norm = psoz / row_max
    ysoz = psoz_norm.mean(axis=0)

    max_chn = int(np.argmax(ysoz))
    correct = 1 if true_onset[max_chn] == 1 else 0
    if correct == 0 and true_onset.sum() <= neighbour_threshold:
        if check_neighborhood(max_chn, true_onset):
            correct = 1

    uncertainty = np.nan_to_num(psoz_norm.var(axis=0), nan=0.0)
    return ysoz, uncertainty, correct


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if roc_auc_score is None or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def _per_channel_auc(probs: np.ndarray, labels: np.ndarray) -> List[float]:
    return [_safe_auc(labels[:, i], probs[:, i]) for i in range(probs.shape[1])]


def _onset_key(onset: np.ndarray) -> str:
    return ",".join(str(int(x)) for x in onset)


def _true_soz_names(onset: np.ndarray) -> List[str]:
    return [TCP_NAMES[i] for i, active in enumerate(onset) if active == 1]


def aggregate_window_predictions(
    probs: np.ndarray,
    event_labels: np.ndarray,
    seizure_labels: np.ndarray,
    event_ids: Sequence[str],
    patient_ids: Sequence[str],
    edf_paths: Sequence[str],
) -> Dict[str, object]:
    """Aggregate positive post-onset windows into event-level prediction sets."""
    grouped: Dict[str, Dict[str, object]] = {}
    for i in range(len(event_ids)):
        if seizure_labels[i] <= 0.5 or event_labels[i].sum() == 0:
            continue
        event_id = str(event_ids[i])
        if event_id not in grouped:
            grouped[event_id] = {
                "psoz": [],
                "label": event_labels[i].astype(np.int64),
                "patient_id": str(patient_ids[i]),
                "edf_path": str(edf_paths[i]),
            }
        grouped[event_id]["psoz"].append(probs[i])

    event_psoz = []
    labels = []
    out_event_ids = []
    out_patient_ids = []
    out_edf_paths = []
    for event_id, item in grouped.items():
        event_psoz.append(np.stack(item["psoz"], axis=0).astype(np.float32))
        labels.append(item["label"])
        out_event_ids.append(event_id)
        out_patient_ids.append(item["patient_id"])
        out_edf_paths.append(item["edf_path"])

    labels_np = (
        np.stack(labels, axis=0).astype(np.int64)
        if labels
        else np.zeros((0, len(TCP_NAMES)), dtype=np.int64)
    )
    return {
        "event_psoz": event_psoz,
        "labels": labels_np,
        "event_ids": out_event_ids,
        "patient_ids": out_patient_ids,
        "edf_paths": out_edf_paths,
    }


def compute_deepsoz_event_metrics(
    event_psoz: Sequence[np.ndarray],
    labels: np.ndarray,
    patient_ids: Sequence[str],
    event_ids: Sequence[str],
    edf_paths: Sequence[str],
    neighbour_threshold: int = 4,
    include_details: bool = True,
) -> Dict[str, object]:
    """Compute DeepSOZ per-channel, seizure-level, and patient-level metrics."""
    n_events = len(event_psoz)
    if n_events == 0:
        empty_metrics = {
            "per_channel_auc": [float("nan")] * len(TCP_NAMES),
            "mean_auc": float("nan"),
            "corr_sz": 0.0,
            "szunc_mean": 0.0,
            "acc_pt_weighted": 0.0,
            "acc_pt_strict": 0.0,
            "acc_pt_lenient": 0.0,
            "n_seizures_evaluated": 0,
            "n_patients_evaluated": 0,
        }
        return {"metrics": empty_metrics, "per_seizure": [], "per_patient": []}

    mean_probs = np.stack([p.mean(axis=0) for p in event_psoz], axis=0)
    ch_aucs = _per_channel_auc(mean_probs, labels)
    valid_aucs = [a for a in ch_aucs if not np.isnan(a)]

    per_seizure = []
    patient_groups: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    patient_onsets: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

    seizure_correct = []
    seizure_unc = []
    for i in range(n_events):
        true_onset = labels[i]
        if true_onset.sum() == 0:
            continue
        ysoz, unc, correct = final_loc(event_psoz[i], true_onset, neighbour_threshold)
        pred_idx = int(np.argmax(ysoz))
        seizure_correct.append(correct)
        seizure_unc.append(float(unc.max()))

        pid = str(patient_ids[i])
        onset_key = _onset_key(true_onset)
        patient_groups[pid][onset_key].append(event_psoz[i])
        patient_onsets[pid][onset_key] = true_onset

        if include_details:
            per_seizure.append({
                "patient_id": pid,
                "event_id": str(event_ids[i]),
                "file_path": os.path.basename(str(edf_paths[i])),
                "correct": int(correct),
                "pred_channel": TCP_NAMES[pred_idx],
                "pred_channel_idx": pred_idx,
                "true_soz": _true_soz_names(true_onset),
                "unc_max": float(unc.max()),
                "n_windows": int(event_psoz[i].shape[0]),
            })

    per_patient = []
    pt_weighted = []
    pt_strict = []
    pt_lenient = []
    for pid, patterns in patient_groups.items():
        pattern_results = []
        for onset_key, psoz_list in patterns.items():
            true_onset = patient_onsets[pid][onset_key]
            stacked = np.concatenate(psoz_list, axis=0)
            ysoz, unc, correct = final_loc(stacked, true_onset, neighbour_threshold)
            pred_idx = int(np.argmax(ysoz))
            pattern_results.append({
                "correct": int(correct),
                "n_seizures": len(psoz_list),
                "pred_channel": TCP_NAMES[pred_idx],
                "true_soz": _true_soz_names(true_onset),
                "unc_max": float(unc.max()),
            })

        total_sz = sum(p["n_seizures"] for p in pattern_results)
        weighted_score = sum(
            p["correct"] * p["n_seizures"] / max(total_sz, 1)
            for p in pattern_results
        )
        correct_weighted = int(weighted_score >= 0.5)
        correct_strict = int(all(p["correct"] for p in pattern_results))
        correct_lenient = int(any(p["correct"] for p in pattern_results))
        pt_weighted.append(correct_weighted)
        pt_strict.append(correct_strict)
        pt_lenient.append(correct_lenient)

        if include_details:
            soz_channels = sorted({
                ch
                for pattern in pattern_results
                for ch in pattern["true_soz"]
            })
            per_patient.append({
                "patient_id": pid,
                "correct_weighted": correct_weighted,
                "correct_strict": correct_strict,
                "correct_lenient": correct_lenient,
                "weighted_score": float(weighted_score),
                "n_seizures": int(total_sz),
                "n_patterns": int(len(pattern_results)),
                "soz_channels": soz_channels,
            })

    metrics = {
        "per_channel_auc": ch_aucs,
        "mean_auc": float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        "corr_sz": float(np.mean(seizure_correct)) if seizure_correct else 0.0,
        "szunc_mean": float(np.mean(seizure_unc)) if seizure_unc else 0.0,
        "acc_pt_weighted": float(np.mean(pt_weighted)) if pt_weighted else 0.0,
        "acc_pt_strict": float(np.mean(pt_strict)) if pt_strict else 0.0,
        "acc_pt_lenient": float(np.mean(pt_lenient)) if pt_lenient else 0.0,
        "n_seizures_evaluated": int(len(seizure_correct)),
        "n_patients_evaluated": int(len(patient_groups)),
    }
    return {
        "metrics": metrics,
        "per_seizure": per_seizure,
        "per_patient": per_patient,
    }


def compute_deepsoz_metrics_from_windows(
    probs: np.ndarray,
    event_labels: np.ndarray,
    seizure_labels: np.ndarray,
    event_ids: Sequence[str],
    patient_ids: Sequence[str],
    edf_paths: Sequence[str],
    neighbour_threshold: int = 4,
    include_details: bool = True,
) -> Dict[str, object]:
    aggregated = aggregate_window_predictions(
        probs=probs,
        event_labels=event_labels,
        seizure_labels=seizure_labels,
        event_ids=event_ids,
        patient_ids=patient_ids,
        edf_paths=edf_paths,
    )
    return compute_deepsoz_event_metrics(
        event_psoz=aggregated["event_psoz"],
        labels=aggregated["labels"],
        patient_ids=aggregated["patient_ids"],
        event_ids=aggregated["event_ids"],
        edf_paths=aggregated["edf_paths"],
        neighbour_threshold=neighbour_threshold,
        include_details=include_details,
    )


@torch.no_grad()
def compute_deepsoz_metrics_mc(
    model,
    loader,
    device,
    mc_samples: int = 20,
    neighbour_threshold: int = 4,
    include_details: bool = True,
) -> Dict[str, object]:
    """Run MC dropout and compute DeepSOZ metrics from positive windows."""
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")

    was_training = model.training
    model.train()
    grouped: Dict[str, Dict[str, object]] = {}

    for batch in loader:
        x = batch["x"].to(device)
        seizure = batch["seizure"].cpu().numpy()
        event_labels = batch["event_soz"].cpu().numpy()
        metas = batch["meta"]

        mc_probs = []
        for _ in range(mc_samples):
            out = model(x)
            mc_probs.append(torch.sigmoid(out["soz_logits"]).cpu().numpy())
        mc_probs = np.stack(mc_probs, axis=1)  # [B, mc_samples, 22]

        for i, meta in enumerate(metas):
            if seizure[i] <= 0.5 or event_labels[i].sum() == 0:
                continue
            event_id = str(meta.get("event_id", ""))
            if event_id not in grouped:
                grouped[event_id] = {
                    "psoz": [],
                    "label": event_labels[i].astype(np.int64),
                    "patient_id": str(meta.get("patient_base") or meta.get("patient_id", "")),
                    "edf_path": str(meta.get("edf_path", "")),
                }
            grouped[event_id]["psoz"].append(mc_probs[i])

    model.train(was_training)

    event_psoz = []
    labels = []
    event_ids = []
    patient_ids = []
    edf_paths = []
    for event_id, item in grouped.items():
        event_psoz.append(np.concatenate(item["psoz"], axis=0).astype(np.float32))
        labels.append(item["label"])
        event_ids.append(event_id)
        patient_ids.append(item["patient_id"])
        edf_paths.append(item["edf_path"])

    labels_np = (
        np.stack(labels, axis=0).astype(np.int64)
        if labels
        else np.zeros((0, len(TCP_NAMES)), dtype=np.int64)
    )
    return compute_deepsoz_event_metrics(
        event_psoz=event_psoz,
        labels=labels_np,
        patient_ids=patient_ids,
        event_ids=event_ids,
        edf_paths=edf_paths,
        neighbour_threshold=neighbour_threshold,
        include_details=include_details,
    )


def save_deepsoz_results(results: Dict[str, Dict[str, object]], output_dir: str | Path) -> None:
    """Save metrics and detail CSVs in the same spirit as baseline/evaluate.py."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode_name, mode_results in results.items():
        if mode_results is None:
            continue
        metrics = mode_results.get("metrics", {})
        metrics_path = output_dir / f"soz_eval_{mode_name}_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        per_seizure = mode_results.get("per_seizure", [])
        if per_seizure:
            with (output_dir / f"soz_eval_{mode_name}_per_seizure.csv").open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as f:
                fieldnames = [
                    "patient_id", "event_id", "file_path", "correct",
                    "pred_channel", "pred_channel_idx", "true_soz",
                    "unc_max", "n_windows",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_seizure:
                    row = dict(row)
                    row["true_soz"] = ";".join(row.get("true_soz", []))
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

        per_patient = mode_results.get("per_patient", [])
        if per_patient:
            with (output_dir / f"soz_eval_{mode_name}_per_patient.csv").open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as f:
                fieldnames = [
                    "patient_id", "correct_weighted", "correct_strict",
                    "correct_lenient", "weighted_score", "n_seizures",
                    "n_patterns", "soz_channels",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_patient:
                    row = dict(row)
                    row["soz_channels"] = ";".join(row.get("soz_channels", []))
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

