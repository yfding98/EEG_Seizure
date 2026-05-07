#!/usr/bin/env python3
"""Onset-aligned lookback dataset for private SOZ MTGAT training."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .soz_manifest import SOZ_COLUMNS, TCP_NAMES, TCP_PAIRS, normalize_electrode
except ImportError:  # Allows: python code/MTGAT/train_soz_mtgat.py
    from soz_manifest import SOZ_COLUMNS, TCP_NAMES, TCP_PAIRS, normalize_electrode


@dataclass(frozen=True)
class WindowIndex:
    row_idx: int
    start: int
    target: int
    is_seizure: int


_ANNOTATION_WARNING_REPORTED: set[tuple[str, str]] = set()


def _read_original_edf_annotations(raw, encoding: str):
    """Re-read EDF TAL annotations before MNE crops them to the data range."""
    try:
        from mne.io.edf.edf import _read_annotations_edf

        edf_info = raw._raw_extras[0]
        tal_idx = edf_info.get("tal_idx", [])
        if len(tal_idx) == 0:
            return None

        idx = np.empty(0, int)
        tal_data = raw._read_segment_file(
            np.empty((0, raw.n_times)),
            idx,
            0,
            0,
            int(raw.n_times),
            np.ones((len(idx), 1)),
            None,
        )
        return _read_annotations_edf(
            tal_data[0],
            ch_names=raw.info["ch_names"],
            encoding=encoding,
        )
    except Exception as exc:  # pragma: no cover - depends on MNE/private EDF internals
        print(
            f"[MTGAT][EDF annotation warning] Could not inspect original EDF "
            f"annotations: {exc}",
            flush=True,
        )
        return None


def _print_out_of_range_annotation_details(
    raw,
    edf_path: Path,
    encoding: str,
    warning_message: str,
    max_items: int = 50,
) -> None:
    """Print file and annotation rows responsible for MNE outside-range warnings."""
    key = (str(edf_path), warning_message)
    if key in _ANNOTATION_WARNING_REPORTED:
        return
    _ANNOTATION_WARNING_REPORTED.add(key)

    data_start = 0.0
    data_stop = float(raw.times[-1] + 1.0 / raw.info["sfreq"])
    print(
        f"[MTGAT][EDF annotation warning] {edf_path}\n"
        f"  MNE warning: {warning_message}\n"
        f"  data_range_sec=[{data_start:.6f}, {data_stop:.6f}], "
        f"sfreq={float(raw.info['sfreq']):.6f}, n_times={raw.n_times}, "
        f"encoding={encoding}",
        flush=True,
    )

    annotations = _read_original_edf_annotations(raw, encoding)
    if annotations is None:
        print("  Could not recover the original annotation rows.", flush=True)
        return

    omitted = []
    for idx, (onset, duration, description, ch_names) in enumerate(
        zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
            annotations.ch_names,
        )
    ):
        duration = 0.0 if np.isnan(duration) else float(duration)
        onset = float(onset)
        end = onset + duration
        if onset > data_stop or end < data_start:
            omitted.append((idx, onset, duration, end, str(description), ch_names))

    if not omitted:
        print(
            "  Original annotations were parsed, but no out-of-range rows were "
            "found with the current MNE crop rule.",
            flush=True,
        )
        return

    print(f"  out_of_range_annotations={len(omitted)}", flush=True)
    for local_idx, item in enumerate(omitted[:max_items]):
        idx, onset, duration, end, description, ch_names = item
        ch_text = ",".join(ch_names) if ch_names else ""
        print(
            f"    [{local_idx:03d}] original_idx={idx}, "
            f"onset={onset:.6f}s, duration={duration:.6f}s, "
            f"end={end:.6f}s, description={description!r}, "
            f"ch_names={ch_text!r}",
            flush=True,
        )
    if len(omitted) > max_items:
        print(
            f"    ... truncated {len(omitted) - max_items} more annotation(s)",
            flush=True,
        )


def _read_raw_edf(edf_path: Path):
    import mne

    last_error = None
    for encoding in ("utf-8", "latin-1"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                raw = mne.io.read_raw_edf(
                    str(edf_path),
                    preload=True,
                    verbose=False,
                    encoding=encoding,
                )
            for warning_item in caught:
                message = str(warning_item.message)
                if (
                    warning_item.category is RuntimeWarning
                    and "annotation(s) that were outside data range" in message
                ):
                    _print_out_of_range_annotation_details(
                        raw,
                        edf_path,
                        encoding,
                        message,
                    )
                else:
                    warnings.warn(
                        warning_item.message,
                        warning_item.category,
                        stacklevel=2,
                    )
            return raw
        except Exception as exc:  # pragma: no cover - depends on local EDFs
            last_error = exc
    raise RuntimeError(f"Could not read EDF {edf_path}: {last_error}")


def _find_channel(ch_names: List[str], target: str) -> int | None:
    target = normalize_electrode(target)
    for idx, name in enumerate(ch_names):
        if normalize_electrode(name) == target:
            return idx
    return None


def _resolve_edf_path(data_root: str | Path, edf_rel: str) -> Path:
    edf_path = Path(str(edf_rel))
    if edf_path.is_absolute():
        return edf_path
    return Path(data_root) / edf_path


def load_event_segment(
    row: pd.Series,
    data_root: str | Path,
    sfreq: float,
    pre_sec: float,
    post_sec: float,
    filter_low: float,
    filter_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one EDF event and return normalized TCP data as (22, T)."""
    edf_path = _resolve_edf_path(data_root, str(row["edf_path"]))
    if not edf_path.is_file():
        raise FileNotFoundError(f"EDF not found: {edf_path}")

    raw = _read_raw_edf(edf_path)
    raw.load_data()
    if filter_low > 0 or filter_high > 0:
        h_freq = min(filter_high, raw.info["sfreq"] / 2.0 - 1.0) if filter_high > 0 else None
        l_freq = filter_low if filter_low > 0 else None
        if h_freq is None or h_freq > (l_freq or 0):
            raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    if abs(float(raw.info["sfreq"]) - sfreq) > 1e-3:
        raw.resample(sfreq, verbose=False)

    data = raw.get_data()
    ch_names = list(raw.ch_names)
    n_samples = data.shape[1]
    bipolar = np.zeros((len(TCP_PAIRS), n_samples), dtype=np.float64)
    mask = np.zeros(len(TCP_PAIRS), dtype=np.float32)

    for idx, (a, b) in enumerate(TCP_PAIRS):
        ia = _find_channel(ch_names, a)
        ib = _find_channel(ch_names, b)
        if ia is None or ib is None:
            continue
        bipolar[idx] = data[ia] - data[ib]
        mask[idx] = 1.0

    pre_samples = int(round(pre_sec * sfreq))
    total_samples = int(round((pre_sec + post_sec) * sfreq))
    onset_sample = int(round(float(row["sz_start"]) * sfreq))
    src_start = onset_sample - pre_samples
    src_end = src_start + total_samples

    segment = np.zeros((len(TCP_PAIRS), total_samples), dtype=np.float64)
    copy_start = max(0, src_start)
    copy_end = min(n_samples, src_end)
    if copy_end > copy_start:
        dst_start = copy_start - src_start
        dst_end = dst_start + (copy_end - copy_start)
        segment[:, dst_start:dst_end] = bipolar[:, copy_start:copy_end]

    baseline = segment[:, :pre_samples]
    mean = baseline.mean(axis=1, keepdims=True)
    std = baseline.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-8)
    segment = (segment - mean) / std
    segment[mask < 0.5] = 0.0
    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    return segment.astype(np.float32), mask.astype(np.float32)


class SOZLookbackDataset(Dataset):
    """Sliding lookback windows centered on seizure onset.

    Each sample uses x[t-lookback:t] to predict whether the next target sample
    belongs to the seizure interval and, if positive, which TCP channels are SOZ.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        data_root: str | Path,
        lookback: int = 200,
        horizon: int = 1,
        stride: int = 20,
        sfreq: float = 200.0,
        pre_sec: float = 15.0,
        post_sec: float = 15.0,
        filter_low: float = 1.0,
        filter_high: float = 50.0,
        cache_size: int = 4,
        max_windows_per_event: int = 0,
    ):
        self.df = manifest_df.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.sfreq = float(sfreq)
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[int, Tuple[np.ndarray, np.ndarray]] = OrderedDict()

        if self.lookback <= 0 or self.horizon <= 0 or self.stride <= 0:
            raise ValueError("lookback, horizon, and stride must be positive")

        self.pre_samples = int(round(self.pre_sec * self.sfreq))
        self.total_samples = int(round((self.pre_sec + self.post_sec) * self.sfreq))
        self.windows = self._build_windows(max_windows_per_event=max_windows_per_event)
        if not self.windows:
            raise ValueError("No lookback windows were created; check lookback/stride/window seconds")

    def _build_windows(self, max_windows_per_event: int = 0) -> List[WindowIndex]:
        all_windows: List[WindowIndex] = []
        last_start = self.total_samples - self.lookback - self.horizon
        if last_start < 0:
            return all_windows

        for row_idx, row in self.df.iterrows():
            event_windows = []
            sz_end_rel = (float(row["sz_end"]) - float(row["sz_start"]) + self.pre_sec) * self.sfreq
            for start in range(0, last_start + 1, self.stride):
                target = start + self.lookback + self.horizon - 1
                is_seizure = int(self.pre_samples <= target < sz_end_rel)
                event_windows.append(WindowIndex(row_idx, start, target, is_seizure))

            if max_windows_per_event and len(event_windows) > max_windows_per_event:
                keep = np.linspace(
                    0,
                    len(event_windows) - 1,
                    num=max_windows_per_event,
                    dtype=int,
                )
                event_windows = [event_windows[int(i)] for i in keep]
            all_windows.extend(event_windows)
        return all_windows

    def __len__(self) -> int:
        return len(self.windows)

    def _get_event(self, row_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if row_idx in self._cache:
            segment, mask = self._cache.pop(row_idx)
            self._cache[row_idx] = (segment, mask)
            return segment, mask

        row = self.df.iloc[row_idx]
        segment, mask = load_event_segment(
            row=row,
            data_root=self.data_root,
            sfreq=self.sfreq,
            pre_sec=self.pre_sec,
            post_sec=self.post_sec,
            filter_low=self.filter_low,
            filter_high=self.filter_high,
        )
        self._cache[row_idx] = (segment, mask)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return segment, mask

    def _row_label(self, row_idx: int) -> np.ndarray:
        row = self.df.iloc[row_idx]
        return row[SOZ_COLUMNS].to_numpy(dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        win = self.windows[idx]
        row = self.df.iloc[win.row_idx]
        segment, mask = self._get_event(win.row_idx)
        x = segment[:, win.start:win.start + self.lookback].T
        seizure = np.float32(win.is_seizure)
        event_soz = self._row_label(win.row_idx)
        soz = event_soz if win.is_seizure else np.zeros_like(event_soz)

        target_rel_sec = (win.target - self.pre_samples) / self.sfreq
        start_rel_sec = (win.start - self.pre_samples) / self.sfreq
        meta = {
            "event_id": str(row.get("event_id", win.row_idx)),
            "patient_id": str(row.get("patient_id", "")),
            "patient_base": str(row.get("patient_base", "")),
            "edf_path": str(row.get("edf_path", "")),
            "sz_start": float(row.get("sz_start", 0.0)),
            "start_rel_sec": float(start_rel_sec),
            "target_rel_sec": float(target_rel_sec),
            "is_seizure": int(win.is_seizure),
        }

        return {
            "x": torch.from_numpy(x).float(),
            "seizure": torch.tensor(seizure, dtype=torch.float32),
            "soz": torch.from_numpy(soz).float(),
            "event_soz": torch.from_numpy(event_soz).float(),
            "mask": torch.from_numpy(mask).float(),
            "meta": meta,
        }

    def label_counts(self) -> Tuple[int, int]:
        n_pos = sum(w.is_seizure for w in self.windows)
        n_neg = len(self.windows) - n_pos
        return int(n_pos), int(n_neg)

    def channel_positive_counts(self) -> np.ndarray:
        counts = np.zeros(len(TCP_NAMES), dtype=np.float64)
        for win in self.windows:
            if win.is_seizure:
                counts += self._row_label(win.row_idx)
        return counts


def collate_soz_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    metas = [item["meta"] for item in batch]
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "seizure": torch.stack([item["seizure"] for item in batch]),
        "soz": torch.stack([item["soz"] for item in batch]),
        "event_soz": torch.stack([item["event_soz"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "meta": metas,
        "event_id": [m["event_id"] for m in metas],
        "patient_id": [m["patient_id"] for m in metas],
    }
