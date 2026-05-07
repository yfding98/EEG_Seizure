#!/usr/bin/env python3
"""Manifest processing for private SOZ onset-localization training.

The private manifest stores one row per seizure event.  This module makes the
SOZ target explicit in a fixed 22-channel TCP order and adds a patient-level
train/val/test split for experiments where the original CSV split is just
"private".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# Fixed order used by private_manifest_clean.csv and code/data_preprocess/eeg_pipeline.py.
TCP_PAIRS: List[Tuple[str, str]] = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
    ("C4", "T4"), ("T4", "A2"),
]
TCP_NAMES = [f"{a}-{b}" for a, b in TCP_PAIRS]
SOZ_COLUMNS = [name.replace("-", "_") for name in TCP_NAMES]
PAIR_TO_INDEX = {name: idx for idx, name in enumerate(TCP_NAMES)}
PAIR_TO_COLUMN = {name: col for name, col in zip(TCP_NAMES, SOZ_COLUMNS)}

CHANNEL_ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}


def normalize_electrode(name: str) -> str:
    """Normalize common EDF/clinical electrode spellings to TCP names."""
    s = str(name).strip().upper()
    for prefix in ("EEG ", "EEG-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in ("-REF", "-LE", "-AR", "-AVG"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return CHANNEL_ALIASES.get(s, s)


def normalize_pair_label(pair: str) -> str | None:
    """Return canonical A-B pair name if it belongs to the fixed TCP order."""
    if pair is None or (isinstance(pair, float) and np.isnan(pair)):
        return None
    text = str(pair).strip().upper().replace("_", "-")
    if not text:
        return None
    parts = [normalize_electrode(p) for p in text.split("-") if p.strip()]
    if len(parts) != 2:
        return None
    name = f"{parts[0]}-{parts[1]}"
    if name in PAIR_TO_INDEX:
        return name
    rev = f"{parts[1]}-{parts[0]}"
    if rev in PAIR_TO_INDEX:
        return rev
    return None


def extract_patient_name(patient_id: str) -> str:
    """Collapse event ids such as NAME_SZ1 to the patient-level id NAME."""
    text = str(patient_id).strip()
    base, sep, suffix = text.rpartition("_")
    if sep and suffix.upper().startswith("SZ"):
        return base
    return text


def _parse_pair_list(value: str) -> Iterable[str]:
    for token in str(value).replace(";", ",").split(","):
        pair = normalize_pair_label(token)
        if pair is not None:
            yield pair


def _parse_onset_channels(value: str) -> List[str]:
    return [
        normalize_electrode(part)
        for part in str(value).replace(",", ";").split(";")
        if str(part).strip()
    ]


def _vector_from_row(
    row: pd.Series,
    infer_from_onset_channels: bool = True,
    always_infer_from_onset: bool = False,
) -> np.ndarray:
    """Build a 22-dim SOZ vector in TCP_NAMES order."""
    label = np.zeros(len(TCP_NAMES), dtype=np.float32)

    for idx, col in enumerate(SOZ_COLUMNS):
        if col not in row:
            continue
        value = pd.to_numeric(row[col], errors="coerce")
        if pd.notna(value) and float(value) > 0:
            label[idx] = 1.0

    for pair in _parse_pair_list(row.get("soz_bipolar", "")):
        label[PAIR_TO_INDEX[pair]] = 1.0

    should_infer = infer_from_onset_channels and (
        always_infer_from_onset or label.sum() == 0
    )
    if should_infer:
        onset_electrodes = set(_parse_onset_channels(row.get("onset_channels", "")))
        for idx, (a, b) in enumerate(TCP_PAIRS):
            if a in onset_electrodes or b in onset_electrodes:
                label[idx] = 1.0

    return label


def _patient_split(
    patient_ids: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    patients = sorted(set(patient_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    n = len(patients)
    if n == 0:
        return {}
    if n == 1:
        return {patients[0]: "train"}
    if n == 2:
        return {patients[0]: "train", patients[1]: "test"}

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    split_map = {}
    for pid in patients[:n_train]:
        split_map[pid] = "train"
    for pid in patients[n_train:n_train + n_val]:
        split_map[pid] = "val"
    for pid in patients[n_train + n_val:]:
        split_map[pid] = "test"
    return split_map


def summarize_manifest(df: pd.DataFrame) -> Dict[str, object]:
    channel_counts = {
        name: int(df[col].sum()) if col in df else 0
        for name, col in zip(TCP_NAMES, SOZ_COLUMNS)
    }
    split_counts = (
        df["soz_split"].value_counts().to_dict()
        if "soz_split" in df
        else {}
    )
    return {
        "rows": int(len(df)),
        "patients": int(df["patient_base"].nunique()) if "patient_base" in df else 0,
        "has_soz_rows": int(df["has_soz"].sum()) if "has_soz" in df else 0,
        "empty_soz_rows": int((df["has_soz"] == 0).sum()) if "has_soz" in df else 0,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "channel_positive_counts": channel_counts,
    }


def prepare_private_soz_manifest(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    infer_from_onset_channels: bool = True,
    always_infer_from_onset: bool = False,
    drop_empty_soz: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load, validate, relabel, split, and optionally save the private manifest."""
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path, encoding="utf-8-sig")

    required = ["patient_id", "edf_path", "duration", "sz_start", "sz_end"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required manifest columns: {missing}")

    for col in ("duration", "sz_start", "sz_end", "sz_duration"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["patient_id", "edf_path", "sz_start", "sz_end"])
    df = df[df["sz_end"] > df["sz_start"]].copy()

    labels = np.stack([
        _vector_from_row(
            row,
            infer_from_onset_channels=infer_from_onset_channels,
            always_infer_from_onset=always_infer_from_onset,
        )
        for _, row in df.iterrows()
    ])
    for idx, col in enumerate(SOZ_COLUMNS):
        df[col] = labels[:, idx].astype(np.int64)

    df["patient_base"] = df["patient_id"].map(extract_patient_name)
    df["event_id"] = [
        f"{pid}__{Path(str(edf)).stem}__{start:.3f}"
        for pid, edf, start in zip(df["patient_id"], df["edf_path"], df["sz_start"])
    ]
    df["n_soz_channels"] = labels.sum(axis=1).astype(np.int64)
    df["has_soz"] = (df["n_soz_channels"] > 0).astype(np.int64)
    df["soz_bipolar_fixed"] = [
        ",".join(name for name, active in zip(TCP_NAMES, vec) if active > 0)
        for vec in labels
    ]

    if drop_empty_soz:
        df = df[df["has_soz"] == 1].copy()

    split_map = _patient_split(
        df["patient_base"].tolist(),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    df["soz_split"] = df["patient_base"].map(split_map).fillna("train")

    summary = summarize_manifest(df)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return df.reset_index(drop=True), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare private SOZ manifest")
    parser.add_argument("--manifest", default="private_manifest_clean.csv")
    parser.add_argument("--output", default="runs/MTGAT_manifest/private_soz_manifest_prepared.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--no_infer_from_onset", action="store_true")
    parser.add_argument("--always_infer_from_onset", action="store_true")
    parser.add_argument("--drop_empty_soz", action="store_true")
    args = parser.parse_args()

    _, summary = prepare_private_soz_manifest(
        args.manifest,
        args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        infer_from_onset_channels=not args.no_infer_from_onset,
        always_infer_from_onset=args.always_infer_from_onset,
        drop_empty_soz=args.drop_empty_soz,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
