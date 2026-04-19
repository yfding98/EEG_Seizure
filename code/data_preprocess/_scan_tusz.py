#!/usr/bin/env python3
"""临时脚本：扫描TUSZ EDF文件通道"""
import mne
mne.set_log_level('ERROR')
from pathlib import Path
import os

root = Path(r'F:\dataset\TUSZ\v2.0.3\edf')
edf_files = []
for dirpath, dirnames, filenames in os.walk(root):
    for fn in filenames:
        if fn.endswith('.edf'):
            edf_files.append(os.path.join(dirpath, fn))
            if len(edf_files) >= 10:
                break
    if len(edf_files) >= 10:
        break

for f in edf_files:
    try:
        raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
        sfreq = raw.info['sfreq']
        n_ch = len(raw.ch_names)
        print(f"File: {os.path.basename(f)}")
        print(f"  sfreq={sfreq}, n_ch={n_ch}")
        print(f"  channels: {raw.ch_names}")
        print()
    except Exception as e:
        print(f"FAILED: {os.path.basename(f)}: {e}")

