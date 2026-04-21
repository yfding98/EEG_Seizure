#!/usr/bin/env python3
"""
Preprocess TUSZ EDF files into .npz for baseline experiments.

Three improvements over v1:
  1. Includes pure background (non-seizure) EDF files
  2. Sliding-window extraction for long seizures
  3. Supports 03_tcp_ar_a montage (missing A1/A2 → ch[8],ch[13] zeroed, mask=0)

Usage:
    python code/baseline/preprocess_tusz.py \
        --manifest tusz_manifest.csv \
        --tusz_root F:/dataset/TUSZ/v2.0.3/edf \
        --output_dir F:/process_dataset/baseline \
        --window_sec 10 --stride_sec 5 --sfreq 200 \
        --max_bckg_windows 3

Output per .npz:
    eeg_data:      (22, window_samples) float32 - TCP bipolar signals
    channel_mask:  (22,)                float32 - 1=valid, 0=missing channel
    soz_labels:    (22,)                float32 - per-channel SOZ labels
    is_seizure:    int64                         - 1=seizure, 0=non-seizure
    patient_id:    str
    split:         str
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

# ── 21 monopolar electrodes (standard 10-20 + A1/A2) ──
MONOPOLAR = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ', 'A1', 'A2',
]

# ── 22 TCP bipolar pairs (TUSZ official order) ──
TCP_PAIRS = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),       # left temporal 0-3
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),       # right temporal 4-7
    ('A1', 'T3'),  ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),       # central 8-13
    ('C4', 'T4'),  ('T4', 'A2'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),       # left parasagittal 14-17
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),       # right parasagittal 18-21
]

SOZ_COLUMNS = [
    'FP1_F7', 'F7_T3', 'T3_T5', 'T5_O1',
    'FP2_F8', 'F8_T4', 'T4_T6', 'T6_O2',
    'A1_T3', 'T3_C3', 'C3_CZ', 'CZ_C4', 'C4_T4', 'T4_A2',
    'FP1_F3', 'F3_C3', 'C3_P3', 'P3_O1',
    'FP2_F4', 'F4_C4', 'C4_P4', 'P4_O2',
]


# ──────────────────────────────────────────────────────────────────────────────
# EDF processing helpers
# ──────────────────────────────────────────────────────────────────────────────

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
    """Compute 22-channel TCP bipolar montage.

    For 03_tcp_ar_a (missing A1/A2): ch[8] and ch[13] will be zero with mask=0.
    """
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


def _normalize(seg):
    """Per-channel z-score normalization."""
    m = seg.mean(axis=1, keepdims=True)
    s = seg.std(axis=1, keepdims=True) + 1e-8
    return ((seg - m) / s).astype(np.float32)


def _load_and_filter(edf_path, sfreq):
    """Read EDF, bandpass filter, resample. Returns mne.Raw."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(1.0, 50.0, verbose=False)
    raw.resample(sfreq, verbose=False)
    return raw


def _sliding_windows(bipolar, mask, start_samp, end_samp, win_len, stride):
    """Generate sliding windows within [start_samp, end_samp)."""
    results = []
    pos = start_samp
    while pos + win_len <= end_samp:
        seg = bipolar[:, pos:pos + win_len]
        results.append((_normalize(seg), mask.copy()))
        pos += stride
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Process seizure events from manifest
# ──────────────────────────────────────────────────────────────────────────────

def _process_seizure_manifest(manifest_path, tusz_root, output_dir,
                              window_sec, stride_sec, sfreq):
    """Process manifest: sliding windows over seizure periods + non-seizure portions."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    win_len = int(window_sec * sfreq)
    stride = int(stride_sec * sfreq)

    # Group seizure events by (edf_path, split) to avoid reading the same EDF multiple times
    edf_groups = defaultdict(list)
    for row in rows:
        edf_rel = row['edf_path'].strip()
        key = edf_rel
        edf_groups[key].append(row)

    stats = {'seizure_windows': 0, 'nonseizure_windows': 0, 'edf_ok': 0, 'edf_fail': 0}

    for edf_rel, events in tqdm(edf_groups.items(), desc='Phase 1: Seizure files'):
        try:
            split = events[0]['split'].strip()
            pid = events[0]['patient_id'].strip()
            edf_path = os.path.join(tusz_root, edf_rel)
            if not os.path.isfile(edf_path):
                stats['edf_fail'] += 1
                continue

            raw = _load_and_filter(edf_path, sfreq)
            bipolar, mask = _compute_bipolar(raw)
            total_samp = bipolar.shape[1]

            # Collect all seizure intervals (in samples) and their SOZ labels
            seizure_intervals = []
            for ev in events:
                sz_s = int(float(ev['sz_start'].strip()) * sfreq)
                sz_e = int(float(ev['sz_end'].strip()) * sfreq)
                sz_s = max(0, sz_s)
                sz_e = min(total_samp, sz_e)
                soz = np.array([int(float(ev[c].strip())) for c in SOZ_COLUMNS], dtype=np.float32)
                seizure_intervals.append((sz_s, sz_e, soz, float(ev['sz_start'].strip())))

            stem = Path(edf_rel).stem

            # ── Seizure windows: sliding window over each seizure period ──
            for sz_s, sz_e, soz, sz_start_sec in seizure_intervals:
                windows = _sliding_windows(bipolar, mask, sz_s, sz_e, win_len, stride)
                if not windows and (sz_e - sz_s) >= win_len // 2:
                    # seizure shorter than window: take what we can, center-padded
                    seg = np.zeros((22, win_len), dtype=np.float64)
                    actual = sz_e - sz_s
                    offset = (win_len - actual) // 2
                    seg[:, offset:offset + actual] = bipolar[:, sz_s:sz_e]
                    windows = [(_normalize(seg), mask.copy())]

                for wi, (data, ch_mask) in enumerate(windows):
                    fname = f"{pid}_{stem}_sz{sz_start_sec:.0f}_w{wi}.npz"
                    np.savez_compressed(
                        os.path.join(output_dir, split, fname),
                        eeg_data=data, channel_mask=ch_mask, soz_labels=soz,
                        is_seizure=np.int64(1), patient_id=pid, split=split,
                    )
                    stats['seizure_windows'] += 1

            # ── Non-seizure windows: from portions outside all seizures ──
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
                    fname = f"{pid}_{stem}_bg{fr_s}_{wi}.npz"
                    np.savez_compressed(
                        os.path.join(output_dir, split, fname),
                        eeg_data=data, channel_mask=ch_mask, soz_labels=soz_zero,
                        is_seizure=np.int64(0), patient_id=pid, split=split,
                    )
                    stats['nonseizure_windows'] += 1

            stats['edf_ok'] += 1
        except Exception:
            stats['edf_fail'] += 1
            traceback.print_exc()

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Process pure background (non-seizure) EDF files
# ──────────────────────────────────────────────────────────────────────────────

def _scan_background_files(tusz_root, manifest_edfs):
    """Scan TUSZ directory for EDF files not in manifest (background-only)."""
    background = []
    tusz_root = Path(tusz_root)
    for split in ['train', 'dev', 'eval']:
        split_dir = tusz_root / split
        if not split_dir.exists():
            continue
        for dp, _, fns in os.walk(split_dir):
            for fn in fns:
                if not fn.endswith('.edf'):
                    continue
                full = Path(dp) / fn
                rel = full.relative_to(tusz_root).as_posix()
                if rel in manifest_edfs:
                    continue
                # check annotation to confirm it's background
                csv_bi = str(full).replace('.edf', '.csv_bi')
                if os.path.exists(csv_bi):
                    with open(csv_bi) as f:
                        if 'seiz' in f.read():
                            continue
                background.append((rel, split))
    return background


def _process_background_files(background_files, tusz_root, output_dir,
                              window_sec, sfreq, max_windows_per_file):
    """Extract non-seizure windows from pure background EDF files."""
    win_len = int(window_sec * sfreq)
    soz_zero = np.zeros(22, dtype=np.float32)
    stats = {'bckg_windows': 0, 'bckg_ok': 0, 'bckg_fail': 0}

    for edf_rel, split in tqdm(background_files, desc='Phase 2: Background files'):
        try:
            edf_path = os.path.join(tusz_root, edf_rel)
            pid = Path(edf_rel).parts[1]  # e.g. train/PATIENT_ID/...
            stem = Path(edf_rel).stem

            raw = _load_and_filter(edf_path, sfreq)
            bipolar, mask = _compute_bipolar(raw)
            total_samp = bipolar.shape[1]

            if total_samp < win_len:
                continue

            # evenly-spaced windows (avoid too many from long files)
            n_possible = (total_samp - win_len) // win_len + 1
            n_windows = min(n_possible, max_windows_per_file)

            if n_windows <= 0:
                continue

            if n_possible <= max_windows_per_file:
                starts = list(range(0, total_samp - win_len + 1, win_len))
            else:
                # evenly space the windows
                spacing = (total_samp - win_len) / (n_windows - 1) if n_windows > 1 else 0
                starts = [int(i * spacing) for i in range(n_windows)]

            for wi, s0 in enumerate(starts):
                seg = bipolar[:, s0:s0 + win_len]
                data = _normalize(seg)
                fname = f"{pid}_{stem}_bckg_w{wi}.npz"
                np.savez_compressed(
                    os.path.join(output_dir, split, fname),
                    eeg_data=data, channel_mask=mask.copy(), soz_labels=soz_zero,
                    is_seizure=np.int64(0), patient_id=pid, split=split,
                )
                stats['bckg_windows'] += 1

            stats['bckg_ok'] += 1
        except Exception:
            stats['bckg_fail'] += 1
            traceback.print_exc()

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(manifest_path, tusz_root, output_dir,
               window_sec=10, stride_sec=5, sfreq=200, max_bckg_windows=3):

    mne.set_log_level('ERROR')
    for split in ['train', 'dev', 'eval']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # ── Phase 1: seizure files from manifest ──
    print('=' * 60)
    print('Phase 1: Processing seizure files from manifest')
    print('=' * 60)
    s1 = _process_seizure_manifest(manifest_path, tusz_root, output_dir,
                                   window_sec, stride_sec, sfreq)
    print(f'  EDF files: {s1["edf_ok"]} ok, {s1["edf_fail"]} failed')
    print(f'  Seizure windows:     {s1["seizure_windows"]}')
    print(f'  Non-seizure windows: {s1["nonseizure_windows"]}')

    # ── Phase 2: pure background files ──
    print()
    print('=' * 60)
    print('Phase 2: Scanning for background-only EDF files')
    print('=' * 60)

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_edfs = set(r['edf_path'].strip() for r in csv.DictReader(f))
    print(f'  Manifest EDF files: {len(manifest_edfs)}')

    background = _scan_background_files(tusz_root, manifest_edfs)
    print(f'  Background EDF files found: {len(background)}')

    s2 = _process_background_files(background, tusz_root, output_dir,
                                    window_sec, sfreq, max_bckg_windows)
    print(f'  Processed: {s2["bckg_ok"]} ok, {s2["bckg_fail"]} failed')
    print(f'  Background windows:  {s2["bckg_windows"]}')

    # ── Summary ──
    total_sz = s1['seizure_windows']
    total_nonsz = s1['nonseizure_windows'] + s2['bckg_windows']
    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'  Total seizure windows:     {total_sz}')
    print(f'  Total non-seizure windows: {total_nonsz}')
    print(f'  Total windows:             {total_sz + total_nonsz}')
    print(f'  Ratio (non-sz / sz):       {total_nonsz / max(total_sz, 1):.1f}')
    print(f'  Output directory: {output_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Preprocess TUSZ for baseline CNN')
    p.add_argument('--manifest', default='tusz_manifest.csv')
    p.add_argument('--tusz_root', default=r'F:\dataset\TUSZ\v2.0.3\edf')
    p.add_argument('--output_dir', default=r'F:\process_dataset\baseline')
    p.add_argument('--window_sec', type=int, default=10,
                   help='Window length in seconds (default: 10)')
    p.add_argument('--stride_sec', type=int, default=5,
                   help='Sliding window stride in seconds for seizure segments (default: 5, i.e. 50%% overlap)')
    p.add_argument('--sfreq', type=int, default=200,
                   help='Target sampling rate (default: 200)')
    p.add_argument('--max_bckg_windows', type=int, default=3,
                   help='Max windows per pure background file (default: 3)')
    args = p.parse_args()
    preprocess(args.manifest, args.tusz_root, args.output_dir,
               args.window_sec, args.stride_sec, args.sfreq, args.max_bckg_windows)
