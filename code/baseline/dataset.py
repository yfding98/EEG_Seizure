#!/usr/bin/env python3
"""PyTorch Dataset for preprocessed TUSZ baseline .npz files."""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TUSZBaselineDataset(Dataset):
    """
    Reads preprocessed .npz files from a split directory.

    Args:
        data_dir:  root of preprocessed data (contains train/dev/eval folders)
        split:     'train', 'dev', or 'eval'
        task:      'detection' (binary) or 'soz' (22-ch multi-label)
    """

    def __init__(self, data_dir: str, split: str, task: str = 'detection'):
        assert task in ('detection', 'soz')
        self.task = task
        split_dir = os.path.join(data_dir, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, '*.npz')))

        if task == 'soz':
            self.files = [f for f in self.files if 'nonseizure' not in os.path.basename(f)]

        if len(self.files) == 0:
            raise FileNotFoundError(f'No .npz files found in {split_dir}')
        print(f'[{split}] Loaded {len(self.files)} samples (task={task})')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        eeg = torch.from_numpy(d['eeg_data'])          # (22, T)
        mask = torch.from_numpy(d['channel_mask'])      # (22,)

        if self.task == 'detection':
            label = torch.tensor(int(d['is_seizure']), dtype=torch.float32)
        else:
            label = torch.from_numpy(d['soz_labels'])   # (22,)

        return eeg, label, mask
