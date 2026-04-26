#!/usr/bin/env python3
"""PyTorch Dataset for preprocessed TUSZ baseline .npz files."""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from baseline.regions import (
    MONOPOLAR_CHANNELS_TUSZ17,
    STANDARD19_TO_TUSZ17,
    STANDARD_19,
    TCP_PAIRS,
    bipolar_to_monopolar_region_labels,
    channel_to_region_labels,
)


def _bipolar_to_monopolar_fallback(eeg_np, mask_np=None):
    """
    Approximate 17-channel monopolar signals from 22 bipolar signals.

    This is a pseudo-inverse fallback for old .npz files that only contain
    eeg_data=(22,T). It cannot recover the true reference-dependent monopolar
    potentials; prefer preprocessing that saves real monopolar_data.
    """
    ch_to_idx = {ch: i for i, ch in enumerate(MONOPOLAR_CHANNELS_TUSZ17)}
    A = np.zeros((len(TCP_PAIRS), len(MONOPOLAR_CHANNELS_TUSZ17)), dtype=np.float32)
    for row, (a, b) in enumerate(TCP_PAIRS):
        if a in ch_to_idx and b in ch_to_idx:
            A[row, ch_to_idx[a]] = 1.0
            A[row, ch_to_idx[b]] = -1.0

    if mask_np is not None:
        valid_rows = (mask_np.astype(np.float32) > 0.5) & (np.abs(A).sum(axis=1) > 0)
    else:
        valid_rows = np.abs(A).sum(axis=1) > 0

    if valid_rows.sum() == 0:
        mono = np.zeros((len(MONOPOLAR_CHANNELS_TUSZ17), eeg_np.shape[1]), dtype=np.float32)
    else:
        mono = np.linalg.pinv(A[valid_rows]).astype(np.float32) @ eeg_np[valid_rows]

    mono_mask = (np.abs(A[valid_rows]).sum(axis=0) > 0).astype(np.float32)
    return mono.astype(np.float32), mono_mask


class TUSZBaselineDataset(Dataset):
    """
    Reads preprocessed .npz files from a split directory.

    Args:
        data_dir:  root of preprocessed data (contains train/dev/eval folders)
        split:     'train', 'dev', or 'eval'
        task:      'detection' (binary), 'soz' (22-ch), or 'soz_region' (5-region)
        return_meta: if True, also return (patient_id, file_path) for evaluation grouping
    """

    def __init__(self, data_dir: str, split: str, task: str = 'detection',
                 return_meta: bool = False, input_mode: str = 'bipolar',
                 allow_bipolar_fallback: bool = True):
        assert task in ('detection', 'soz', 'soz_region')
        assert input_mode in ('bipolar', 'monopolar')
        self.task = task
        self.input_mode = input_mode
        self.allow_bipolar_fallback = allow_bipolar_fallback
        self.return_meta = return_meta
        split_dir = os.path.join(data_dir, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, '*.npz')))

        if task in ('soz', 'soz_region'):
            self.files = [f for f in self.files
                          if 'nonseizure' not in os.path.basename(f)
                          and '_bckg_' not in os.path.basename(f)]

        if len(self.files) == 0:
            raise FileNotFoundError(f'No .npz files found in {split_dir}')
        print(f'[{split}] Loaded {len(self.files)} samples '
              f'(task={task}, input_mode={input_mode})')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        if self.input_mode == 'monopolar':
            eeg_np, mask_np = self._load_monopolar(d)
            eeg = torch.from_numpy(eeg_np)
            mask = torch.from_numpy(mask_np)
        else:
            eeg = torch.from_numpy(d['eeg_data'])          # (22, T)
            mask = torch.from_numpy(d['channel_mask'])      # (22,)

        if self.task == 'detection':
            label = torch.tensor(int(d['is_seizure']), dtype=torch.float32)
        else:
            label = torch.from_numpy(d['soz_labels'])   # (22,)
            if self.task == 'soz_region':
                if self.input_mode == 'monopolar':
                    label = bipolar_to_monopolar_region_labels(label)  # (5,)
                else:
                    label = channel_to_region_labels(label)  # (5,)

        if self.return_meta:
            pid = str(d['patient_id'])
            fpath = self.files[idx]
            return eeg, label, mask, pid, fpath

        return eeg, label, mask

    def _load_monopolar(self, d):
        for key in ('monopolar_data', 'eeg_monopolar', 'x_monopolar'):
            if key in d:
                mono = np.asarray(d[key], dtype=np.float32)
                if mono.shape[0] == len(STANDARD_19):
                    mono = mono[STANDARD19_TO_TUSZ17]
                elif mono.shape[0] != len(MONOPOLAR_CHANNELS_TUSZ17):
                    raise ValueError(
                        f'Unexpected {key} shape: {mono.shape}; expected 17 or 19 channels'
                    )

                mask_key = f'{key}_mask'
                if mask_key in d:
                    mono_mask = np.asarray(d[mask_key], dtype=np.float32)
                    if mono_mask.shape[0] == len(STANDARD_19):
                        mono_mask = mono_mask[STANDARD19_TO_TUSZ17]
                elif 'monopolar_mask' in d:
                    mono_mask = np.asarray(d['monopolar_mask'], dtype=np.float32)
                    if mono_mask.shape[0] == len(STANDARD_19):
                        mono_mask = mono_mask[STANDARD19_TO_TUSZ17]
                else:
                    mono_mask = (np.abs(mono).sum(axis=1) > 0).astype(np.float32)
                return mono, mono_mask

        if not self.allow_bipolar_fallback:
            raise KeyError(
                'input_mode=monopolar requires monopolar_data/eeg_monopolar/x_monopolar '
                'in the .npz file, or allow_bipolar_fallback=True'
            )
        channel_mask = d['channel_mask'] if 'channel_mask' in d else None
        return _bipolar_to_monopolar_fallback(d['eeg_data'], channel_mask)
