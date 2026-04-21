#!/usr/bin/env python3
"""
Preprocess TUSZ EDF files into .npz for baseline experiments.

Usage:
    python -m code.baseline.preprocess_tusz \
        --manifest tusz_manifest.csv \
        --tusz_root F:/dataset/TUSZ/v2.0.3/edf \
        --output_dir F:/process_dataset/baseline \
        --window_sec 10 --sfreq 200

Output per .npz:
    eeg_data:      (22, 2000)  float32  - TCP bipolar signals
    channel_mask:  (22,)       float32  - channel validity
    soz_labels:    (22,)       float32  - per-channel SOZ labels
    is_seizure:    int64       - 1=seizure, 0=non-seizure
    patient_id:    str
    split:         str
"""

import argparse
import csv
import os
import traceback
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

# ── 21 monopolar electrodes (standard 10-20 + A1/A2) ──
MONOPOLAR = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ', 'A1', 'A2',
]

# ── 22 TCP bipolar pairs (TUSZ official order) ──
TCP_PAIRS = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('A1', 'T3'),  ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'), ('C4', 'T4'), ('T4', 'A2'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
]

SOZ_COLUMNS = [
    'FP1_F7', 'F7_T3', 'T3_T5', 'T5_O1',
    'FP2_F8', 'F8_T4', 'T4_T6', 'T6_O2',
    'A1_T3', 'T3_C3', 'C3_CZ', 'CZ_C4', 'C4_T4', 'T4_A2',
    'FP1_F3', 'F3_C3', 'C3_P3', 'P3_O1',
    'FP2_F4', 'F4_C4', 'C4_P4', 'P4_O2',
]


def _find_channel(raw, name):
    """Match monopolar channel name in EDF (handles TUSZ naming: EEG XX-REF / EEG XX-LE)."""
    name_up = name.upper()
    for ch in raw.ch_names:
        norm = ch.upper().replace(' ', '')
        for suffix in ['-REF', '-LE', '-AR']:
            norm = norm.replace(suffix, '')
        if norm.startswith('EEG'):
            norm = norm[3:]
        if norm == name_up:
            return ch
    return None


def _compute_bipolar(raw):
    """Compute 22-channel TCP bipolar montage. Returns (22, n_samples), mask (22,)."""
    data = raw.get_data()
    ch_data = {}
    for name in MONOPOLAR:
        found = _find_channel(raw, name)
        if found is not None:
            ch_data[name] = data[raw.ch_names.index(found)]

    n_samples = data.shape[1]
    bipolar = np.zeros((22, n_samples), dtype=np.float64)
    mask = np.ones(22, dtype=np.float32)

    for i, (anode, cathode) in enumerate(TCP_PAIRS):
        if anode in ch_data and cathode in ch_data:
            bipolar[i] = ch_data[anode] - ch_data[cathode]
        else:
            mask[i] = 0.0

    return bipolar, mask


def _extract_windows(bipolar, mask, sz_start, sz_end, sfreq, window_sec):
    """Extract one seizure window and (if possible) one non-seizure window."""
    win_len = int(window_sec * sfreq)
    total = bipolar.shape[1]
    results = []

    def _normalize(seg):
        m = seg.mean(axis=1, keepdims=True)
        s = seg.std(axis=1, keepdims=True) + 1e-8
        return ((seg - m) / s).astype(np.float32)

    # ── seizure window: centered on seizure midpoint ──
    mid = int((sz_start + sz_end) / 2 * sfreq)
    s0 = max(0, mid - win_len // 2)
    s1 = s0 + win_len
    if s1 > total:
        s1 = total
        s0 = max(0, s1 - win_len)
    if s1 - s0 == win_len:
        results.append(('seizure', _normalize(bipolar[:, s0:s1]), mask.copy()))

    # ── non-seizure window: 30s gap before seizure onset ──
    gap = int(30 * sfreq)
    ns_end = int(sz_start * sfreq) - gap
    ns_start = ns_end - win_len
    if ns_start >= 0:
        results.append(('nonseizure', _normalize(bipolar[:, ns_start:ns_end]), mask.copy()))
    else:
        # try after seizure end
        ns_start = int(sz_end * sfreq) + gap
        ns_end = ns_start + win_len
        if ns_end <= total:
            results.append(('nonseizure', _normalize(bipolar[:, ns_start:ns_end]), mask.copy()))

    return results


def preprocess(manifest_path, tusz_root, output_dir, window_sec=10, sfreq=200):
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'dev', 'eval']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    mne.set_log_level('ERROR')
    ok, fail = 0, 0

    for row in tqdm(rows, desc='Preprocessing TUSZ'):
        try:
            pid = row['patient_id'].strip()
            edf_rel = row['edf_path'].strip()
            split = row['split'].strip()
            sz_start = float(row['sz_start'].strip())
            sz_end = float(row['sz_end'].strip())

            # skip 03_tcp_ar_a (only 20 channels)
            if '03_tcp_ar_a' in edf_rel:
                continue

            soz = np.array([int(float(row[c].strip())) for c in SOZ_COLUMNS], dtype=np.float32)

            edf_path = os.path.join(tusz_root, edf_rel)
            if not os.path.isfile(edf_path):
                fail += 1
                continue

            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.filter(1.0, 50.0, verbose=False)
            raw.resample(sfreq, verbose=False)

            bipolar, mask = _compute_bipolar(raw)
            windows = _extract_windows(bipolar, mask, sz_start, sz_end, sfreq, window_sec)

            stem = Path(edf_rel).stem
            for label_type, data, ch_mask in windows:
                fname = f"{pid}_{stem}_{sz_start:.0f}_{label_type}.npz"
                np.savez_compressed(
                    os.path.join(output_dir, split, fname),
                    eeg_data=data,
                    channel_mask=ch_mask,
                    soz_labels=soz,
                    is_seizure=np.int64(1 if label_type == 'seizure' else 0),
                    patient_id=pid,
                    split=split,
                )
            ok += 1
        except Exception:
            fail += 1
            traceback.print_exc()

    print(f'\nDone: {ok} succeeded, {fail} failed out of {len(rows)} rows')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Preprocess TUSZ for baseline CNN')
    p.add_argument('--manifest', default='tusz_manifest.csv')
    p.add_argument('--tusz_root', default=r'F:\dataset\TUSZ\v2.0.3\edf')
    p.add_argument('--output_dir', default=r'F:\process_dataset\baseline')
    p.add_argument('--window_sec', type=int, default=10)
    p.add_argument('--sfreq', type=int, default=200)
    args = p.parse_args()
    preprocess(args.manifest, args.tusz_root, args.output_dir, args.window_sec, args.sfreq)
