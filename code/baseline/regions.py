#!/usr/bin/env python3
"""Brain-region definitions and label helpers for SOZ localization."""


REGION_NAMES = [
    'left_frontal',
    'right_frontal',
    'parietal',
    'left_temporal',
    'right_temporal',
]

STANDARD_19 = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ',
]

# TUSZ TCP recordings used here do not reliably expose FZ/PZ as raw monopolar
# signals. This 17-channel order is the default for monopolar region models.
MONOPOLAR_CHANNELS_TUSZ17 = [
    ch for ch in STANDARD_19
    if ch not in ('FZ', 'PZ')
]


# Keep this local instead of importing preprocess_tusz.TCP_PAIRS, because that
# module imports mne and should not be required just to instantiate a model.
TCP_PAIRS = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
    ('C4', 'T4'), ('T4', 'A2'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
]


# Monopolar grouping from the private-data annotation plan. The current baseline
# inputs are TCP bipolar signals, so this is kept as reference for future use.
MONOPOLAR_REGIONS = {
    'left_frontal': ['FP1', 'F7', 'F3'],
    'right_frontal': ['FP2', 'F8', 'F4'],
    'parietal': ['FZ', 'CZ', 'C3', 'C4', 'P3', 'PZ', 'P4'],
    'left_temporal': ['F7', 'T3', 'T5', 'O1', 'C3', 'P3'],
    'right_temporal': ['F8', 'T4', 'T6', 'O2', 'C4', 'P4'],
}

MONOPOLAR_REGION_INDICES_17 = {
    region: [
        MONOPOLAR_CHANNELS_TUSZ17.index(ch)
        for ch in channels
        if ch in MONOPOLAR_CHANNELS_TUSZ17
    ]
    for region, channels in MONOPOLAR_REGIONS.items()
}

STANDARD19_TO_TUSZ17 = [
    STANDARD_19.index(ch)
    for ch in MONOPOLAR_CHANNELS_TUSZ17
]


BIPOLAR_REGIONS = {
    'left_frontal': ['FP1-F7', 'FP1-F3', 'F7-F3', 'F3-FZ'],
    'left_temporal': ['F7-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'T5-P3'],
    'parietal': ['FZ-CZ', 'C3-CZ', 'P3-PZ', 'CZ-PZ', 'CZ-C4', 'PZ-P4'],
    'right_frontal': ['FP2-F4', 'FP2-F8', 'F4-F8', 'FZ-F4'],
    'right_temporal': ['F8-T4', 'C4-T4', 'T4-T6', 'P4-T6', 'T6-O2'],
}


TCP_NAMES = [f'{a}-{b}' for a, b in TCP_PAIRS]
TCP_NAME_TO_INDEX = {name: i for i, name in enumerate(TCP_NAMES)}


def _region_indices():
    """Map requested bipolar region names to available 22-TCP channel indices."""
    out = {}
    for region in REGION_NAMES:
        out[region] = [
            TCP_NAME_TO_INDEX[name]
            for name in BIPOLAR_REGIONS[region]
            if name in TCP_NAME_TO_INDEX
        ]
    return out


BIPOLAR_REGION_INDICES = _region_indices()


def channel_to_region_labels(channel_labels):
    """Convert 22-channel SOZ labels to 5 brain-region labels by max pooling."""
    import torch

    labels = []
    for region in REGION_NAMES:
        idx = BIPOLAR_REGION_INDICES[region]
        if not idx:
            labels.append(channel_labels.new_tensor(0.0))
        else:
            labels.append(channel_labels[idx].max())
    return torch.stack(labels)


def bipolar_to_monopolar_labels(channel_labels, include_fz_pz=False):
    """Map 22 bipolar SOZ labels to 17/19 monopolar electrode labels."""
    import torch

    mono = []
    for ch in STANDARD_19:
        idx = [
            i for i, pair in enumerate(TCP_PAIRS)
            if ch in pair
        ]
        if idx:
            mono.append(channel_labels[idx].max())
        else:
            mono.append(channel_labels.new_tensor(0.0))
    mono = torch.stack(mono)
    if include_fz_pz:
        return mono
    return mono[STANDARD19_TO_TUSZ17]


def monopolar_to_region_labels(monopolar_labels):
    """Convert 17-channel monopolar SOZ labels to 5 brain-region labels."""
    import torch

    labels = []
    for region in REGION_NAMES:
        idx = MONOPOLAR_REGION_INDICES_17[region]
        if not idx:
            labels.append(monopolar_labels.new_tensor(0.0))
        else:
            labels.append(monopolar_labels[idx].max())
    return torch.stack(labels)


def bipolar_to_monopolar_region_labels(channel_labels):
    """Convert 22-channel bipolar SOZ labels to 5 monopolar-region labels."""
    return monopolar_to_region_labels(
        bipolar_to_monopolar_labels(channel_labels, include_fz_pz=False)
    )
