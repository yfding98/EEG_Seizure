#!/usr/bin/env python3
"""
Preprocess private EDF files into .npz for baseline experiments.

Reuses the same TCP bipolar pipeline as preprocess_tusz.py.
Applies patient-level train/val/test split (70/15/15).

Usage:
    python code/baseline/preprocess_private.py \
        --manifest private_manifest_clean.csv \
        --data_root E:/DataSet/EEG/EEG\ dataset_SUAT \
        --output_dir F:/process_dataset/baseline_private \
        --window_sec 10 --stride_sec 5 --sfreq 200 --seed 42
"""

import argparse
import csv
import os
import traceback
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

from baseline.preprocess_tusz import (
    TCP_PAIRS, SOZ_COLUMNS, _compute_bipolar, _normalize, _load_and_filter,
    _sliding_windows,
)


def _patient_split(patient_ids, seed=42, train_ratio=0.70, val_ratio=0.15):
    """Split patients into train/val/test sets."""
    rng = np.random.RandomState(seed)
    patients = sorted(set(patient_ids))
    rng.shuffle(patients)

    n = len(patients)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_pts = set(patients[:n_train])
    val_pts = set(patients[n_train:n_train + n_val])
    test_pts = set(patients[n_train + n_val:])

    print(f'Patient split (seed={seed}): '
          f'train={len(train_pts)}, val={len(val_pts)}, test={len(test_pts)}')
    return train_pts, val_pts, test_pts


def _extract_patient_name(patient_id):
    """Extract base patient name from patient_id (e.g. '刘娟_SZ1' -> '刘娟')."""
    parts = patient_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].upper().startswith('SZ'):
        return parts[0]
    return patient_id


def preprocess_private(manifest_path, data_root, output_dir,
                       window_sec=10, stride_sec=5, sfreq=200, seed=42):
    mne.set_log_level('ERROR')

    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Patient-level split
    patient_names = [_extract_patient_name(r['patient_id'].strip()) for r in rows]
    train_pts, val_pts, test_pts = _patient_split(patient_names, seed=seed)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    win_len = int(window_sec * sfreq)
    stride = int(stride_sec * sfreq)

    # Group by EDF file
    edf_groups = defaultdict(list)
    for row in rows:
        edf_groups[row['edf_path'].strip()].append(row)

    stats = {'seizure_windows': 0, 'nonseizure_windows': 0, 'ok': 0, 'fail': 0}

    for edf_rel, events in tqdm(edf_groups.items(), desc='Processing private data'):
        try:
            pid_raw = events[0]['patient_id'].strip()
            patient_name = _extract_patient_name(pid_raw)

            if patient_name in train_pts:
                split = 'train'
            elif patient_name in val_pts:
                split = 'val'
            else:
                split = 'test'

            edf_path = os.path.join(data_root, edf_rel)
            if not os.path.isfile(edf_path):
                stats['fail'] += 1
                continue

            raw = _load_and_filter(edf_path, sfreq)
            bipolar, mask = _compute_bipolar(raw)
            total_samp = bipolar.shape[1]

            seizure_intervals = []
            for ev in events:
                sz_s = max(0, int(float(ev['sz_start'].strip()) * sfreq))
                sz_e = min(total_samp, int(float(ev['sz_end'].strip()) * sfreq))
                soz = np.array([int(float(ev[c].strip())) for c in SOZ_COLUMNS],
                               dtype=np.float32)
                seizure_intervals.append((sz_s, sz_e, soz, float(ev['sz_start'].strip())))

            stem = Path(edf_rel).stem
            # Use patient_name as prefix for cleaner filenames
            safe_pid = patient_name.replace('/', '_').replace('\\', '_')

            # Seizure windows
            for sz_s, sz_e, soz, sz_start_sec in seizure_intervals:
                windows = _sliding_windows(bipolar, mask, sz_s, sz_e, win_len, stride)
                if not windows and (sz_e - sz_s) >= win_len // 2:
                    seg = np.zeros((22, win_len), dtype=np.float64)
                    actual = sz_e - sz_s
                    offset = (win_len - actual) // 2
                    seg[:, offset:offset + actual] = bipolar[:, sz_s:sz_e]
                    windows = [(_normalize(seg), mask.copy())]

                for wi, (data, ch_mask) in enumerate(windows):
                    fname = f"{safe_pid}_{stem}_sz{sz_start_sec:.0f}_w{wi}.npz"
                    np.savez_compressed(
                        os.path.join(output_dir, split, fname),
                        eeg_data=data, channel_mask=ch_mask, soz_labels=soz,
                        is_seizure=np.int64(1), patient_id=patient_name, split=split,
                    )
                    stats['seizure_windows'] += 1

            # Non-seizure windows from free ranges
            seizure_intervals_sorted = sorted(seizure_intervals, key=lambda x: x[0])
            free_ranges = []
            prev_end = 0
            for sz_s, sz_e, _, _ in seizure_intervals_sorted:
                if sz_s > prev_end:
                    free_ranges.append((prev_end, sz_s))
                prev_end = max(prev_end, sz_e)
            if prev_end < total_samp:
                free_ranges.append((prev_end, total_samp))

            soz_zero = np.zeros(22, dtype=np.float32)
            for fr_s, fr_e in free_ranges:
                windows = _sliding_windows(bipolar, mask, fr_s, fr_e, win_len, stride)
                for wi, (data, ch_mask) in enumerate(windows):
                    fname = f"{safe_pid}_{stem}_bg{fr_s}_{wi}.npz"
                    np.savez_compressed(
                        os.path.join(output_dir, split, fname),
                        eeg_data=data, channel_mask=ch_mask, soz_labels=soz_zero,
                        is_seizure=np.int64(0), patient_id=patient_name, split=split,
                    )
                    stats['nonseizure_windows'] += 1

            stats['ok'] += 1
        except Exception:
            stats['fail'] += 1
            traceback.print_exc()

    # Print summary
    print(f'\nDone: {stats["ok"]} files ok, {stats["fail"]} failed')
    print(f'  Seizure windows:     {stats["seizure_windows"]}')
    print(f'  Non-seizure windows: {stats["nonseizure_windows"]}')
    for split in ['train', 'val', 'test']:
        import glob
        n = len(glob.glob(os.path.join(output_dir, split, '*.npz')))
        print(f'  {split}: {n} files')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Preprocess private EEG data')
    p.add_argument('--manifest', default='private_manifest_clean.csv')
    p.add_argument('--data_root', default=r'E:\DataSet\EEG\EEG dataset_SUAT')
    p.add_argument('--output_dir', default=r'F:\process_dataset\baseline_private')
    p.add_argument('--window_sec', type=int, default=10)
    p.add_argument('--stride_sec', type=int, default=5)
    p.add_argument('--sfreq', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    preprocess_private(args.manifest, args.data_root, args.output_dir,
                       args.window_sec, args.stride_sec, args.sfreq, args.seed)
