#!/usr/bin/env python3
"""Train MTGAT for private-data seizure onset and SOZ localization.

Default window design:
  - Load [sz_start - 15s, sz_start + 15s].
  - Normalize every channel by the pre-onset 15s baseline.
  - Use x[t-lookback:t] to predict the next target point.
  - If that target point is after sz_start, the SOZ target is the event-level
    22-channel SOZ vector; otherwise it is all zeros.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from deepsoz_eval import (
        compute_deepsoz_metrics_from_windows,
        compute_deepsoz_metrics_mc,
        save_deepsoz_results,
    )
    from model import MTGATSOZ
    from soz_dataset import SOZLookbackDataset, collate_soz_batch
    from soz_manifest import SOZ_COLUMNS, TCP_NAMES, prepare_private_soz_manifest
else:
    from .deepsoz_eval import (
        compute_deepsoz_metrics_from_windows,
        compute_deepsoz_metrics_mc,
        save_deepsoz_results,
    )
    from .model import MTGATSOZ
    from .soz_dataset import SOZLookbackDataset, collate_soz_batch
    from .soz_manifest import SOZ_COLUMNS, TCP_NAMES, prepare_private_soz_manifest

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
except ImportError:  # pragma: no cover
    accuracy_score = f1_score = precision_score = recall_score = roc_auc_score = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_jsonable(value):
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def masked_soz_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    seizure_label: torch.Tensor,
    channel_pos_weight: torch.Tensor,
    background_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=channel_pos_weight,
        reduction="none",
    )
    sample_weight = torch.where(
        seizure_label > 0.5,
        torch.ones_like(seizure_label),
        torch.full_like(seizure_label, float(background_weight)),
    ).unsqueeze(1)
    weights = channel_mask * sample_weight
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def train_one_epoch(
    model: MTGATSOZ,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seizure_pos_weight: torch.Tensor,
    channel_pos_weight: torch.Tensor,
    seizure_loss_weight: float,
    soz_loss_weight: float,
    background_soz_weight: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    totals = defaultdict(float)
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        seizure = batch["seizure"].to(device)
        soz = batch["soz"].to(device)
        mask = batch["mask"].to(device)

        out = model(x)
        det_loss = F.binary_cross_entropy_with_logits(
            out["seizure_logit"],
            seizure,
            pos_weight=seizure_pos_weight,
        )
        soz_loss = masked_soz_loss(
            out["soz_logits"],
            soz,
            mask,
            seizure,
            channel_pos_weight=channel_pos_weight,
            background_weight=background_soz_weight,
        )
        loss = seizure_loss_weight * det_loss + soz_loss_weight * soz_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = x.size(0)
        totals["loss"] += float(loss.item()) * bs
        totals["det_loss"] += float(det_loss.item()) * bs
        totals["soz_loss"] += float(soz_loss.item()) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if roc_auc_score is None or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def compute_metrics(
    seizure_probs: np.ndarray,
    seizure_labels: np.ndarray,
    soz_probs: np.ndarray,
    soz_labels: np.ndarray,
    event_soz_labels: np.ndarray,
    event_ids: List[str],
    patient_ids: List[str],
    edf_paths: List[str],
    neighbour_threshold: int = 4,
    include_deepsoz_details: bool = False,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    det_pred = (seizure_probs >= 0.5).astype(np.int64)
    det_true = seizure_labels.astype(np.int64)
    if accuracy_score is not None:
        metrics["det_accuracy"] = float(accuracy_score(det_true, det_pred))
        metrics["det_f1"] = float(f1_score(det_true, det_pred, zero_division=0))
        metrics["det_precision"] = float(precision_score(det_true, det_pred, zero_division=0))
        metrics["det_recall"] = float(recall_score(det_true, det_pred, zero_division=0))
    metrics["det_auc"] = _safe_auc(det_true, seizure_probs)

    positive = (seizure_labels > 0.5) & (soz_labels.sum(axis=1) > 0)
    metrics["n_positive_soz_windows"] = int(positive.sum())
    if positive.any():
        p = soz_probs[positive]
        y = soz_labels[positive].astype(np.int64)
        pred = (p >= 0.5).astype(np.int64)
        per_auc = [_safe_auc(y[:, i], p[:, i]) for i in range(p.shape[1])]
        valid_auc = [v for v in per_auc if not np.isnan(v)]
        metrics["soz_mean_auc"] = float(np.mean(valid_auc)) if valid_auc else float("nan")
        metrics["soz_per_channel_auc"] = per_auc
        if f1_score is not None:
            metrics["soz_f1_micro"] = float(f1_score(y.ravel(), pred.ravel(), zero_division=0))
            metrics["soz_f1_macro"] = float(f1_score(y, pred, average="macro", zero_division=0))
        top1 = np.argmax(p, axis=1)
        metrics["soz_top1_hit"] = float(np.mean([y[i, ch] > 0 for i, ch in enumerate(top1)]))

        event_groups: Dict[str, List[int]] = defaultdict(list)
        pos_indices = np.where(positive)[0]
        for local_idx, global_idx in enumerate(pos_indices):
            event_groups[event_ids[global_idx]].append(local_idx)
        event_hits = []
        for idxs in event_groups.values():
            mean_prob = p[idxs].mean(axis=0)
            label = y[idxs].max(axis=0)
            event_hits.append(float(label[int(np.argmax(mean_prob))] > 0))
        metrics["event_top1_hit"] = float(np.mean(event_hits)) if event_hits else float("nan")
        metrics["n_eval_events"] = int(len(event_groups))
    else:
        metrics["soz_mean_auc"] = float("nan")
        metrics["soz_per_channel_auc"] = [float("nan")] * len(TCP_NAMES)
        metrics["soz_f1_micro"] = float("nan")
        metrics["soz_f1_macro"] = float("nan")
        metrics["soz_top1_hit"] = float("nan")
        metrics["event_top1_hit"] = float("nan")
        metrics["n_eval_events"] = 0

    deepsoz = compute_deepsoz_metrics_from_windows(
        probs=soz_probs,
        event_labels=event_soz_labels,
        seizure_labels=seizure_labels,
        event_ids=event_ids,
        patient_ids=patient_ids,
        edf_paths=edf_paths,
        neighbour_threshold=neighbour_threshold,
        include_details=include_deepsoz_details,
    )
    for key, value in deepsoz["metrics"].items():
        if key == "per_channel_auc":
            metrics["deepsoz_per_channel_auc"] = value
        else:
            metrics[f"deepsoz_{key}"] = value
    if include_deepsoz_details:
        metrics["deepsoz_details"] = deepsoz
    return metrics


@torch.no_grad()
def evaluate(
    model: MTGATSOZ,
    loader: DataLoader,
    device: torch.device,
    seizure_pos_weight: torch.Tensor,
    channel_pos_weight: torch.Tensor,
    seizure_loss_weight: float,
    soz_loss_weight: float,
    background_soz_weight: float,
    neighbour_threshold: int = 4,
    include_deepsoz_details: bool = False,
) -> Dict[str, object]:
    model.eval()
    totals = defaultdict(float)
    n = 0
    seizure_probs, seizure_labels = [], []
    soz_probs, soz_labels, event_soz_labels = [], [], []
    event_ids, patient_ids, edf_paths = [], [], []

    for batch in loader:
        x = batch["x"].to(device)
        seizure = batch["seizure"].to(device)
        soz = batch["soz"].to(device)
        mask = batch["mask"].to(device)
        out = model(x)

        det_loss = F.binary_cross_entropy_with_logits(
            out["seizure_logit"],
            seizure,
            pos_weight=seizure_pos_weight,
        )
        soz_loss = masked_soz_loss(
            out["soz_logits"],
            soz,
            mask,
            seizure,
            channel_pos_weight=channel_pos_weight,
            background_weight=background_soz_weight,
        )
        loss = seizure_loss_weight * det_loss + soz_loss_weight * soz_loss

        bs = x.size(0)
        totals["loss"] += float(loss.item()) * bs
        totals["det_loss"] += float(det_loss.item()) * bs
        totals["soz_loss"] += float(soz_loss.item()) * bs
        n += bs

        seizure_probs.append(torch.sigmoid(out["seizure_logit"]).cpu().numpy())
        seizure_labels.append(seizure.cpu().numpy())
        soz_probs.append(torch.sigmoid(out["soz_logits"]).cpu().numpy())
        soz_labels.append(soz.cpu().numpy())
        event_soz_labels.append(batch["event_soz"].cpu().numpy())
        event_ids.extend(batch["event_id"])
        patient_ids.extend([
            str(meta.get("patient_base") or meta.get("patient_id", ""))
            for meta in batch["meta"]
        ])
        edf_paths.extend([str(meta.get("edf_path", "")) for meta in batch["meta"]])

    base = {k: v / max(n, 1) for k, v in totals.items()}
    metrics = compute_metrics(
        np.concatenate(seizure_probs),
        np.concatenate(seizure_labels),
        np.concatenate(soz_probs),
        np.concatenate(soz_labels),
        np.concatenate(event_soz_labels),
        event_ids,
        patient_ids,
        edf_paths,
        neighbour_threshold=neighbour_threshold,
        include_deepsoz_details=include_deepsoz_details,
    )
    return {**base, **metrics}


def build_loader(dataset: SOZLookbackDataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= args.batch_size,
        collate_fn=collate_soz_batch,
    )


def build_datasets(df, args) -> Dict[str, SOZLookbackDataset]:
    datasets = {}
    for split in ("train", "val", "test"):
        sdf = df[df["soz_split"] == split].reset_index(drop=True)
        if len(sdf) == 0:
            continue
        datasets[split] = SOZLookbackDataset(
            sdf,
            data_root=args.data_root,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
            sfreq=args.sfreq,
            pre_sec=args.pre_sec,
            post_sec=args.post_sec,
            filter_low=args.filter_low,
            filter_high=args.filter_high,
            cache_size=args.cache_size,
            max_windows_per_event=args.max_windows_per_event,
        )
    return datasets


def compute_loss_weights(train_ds: SOZLookbackDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n_pos, n_neg = train_ds.label_counts()
    seizure_pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)],
        dtype=torch.float32,
        device=device,
    )

    pos_counts = train_ds.channel_positive_counts()
    n_positive_windows = max(n_pos, 1)
    neg_counts = np.maximum(n_positive_windows - pos_counts, 1.0)
    channel_weights = neg_counts / np.maximum(pos_counts, 1.0)
    channel_weights = np.clip(channel_weights, 1.0, 25.0)
    return seizure_pos_weight, torch.tensor(channel_weights, dtype=torch.float32, device=device)


def print_split_info(datasets: Dict[str, SOZLookbackDataset]) -> None:
    for split, ds in datasets.items():
        n_pos, n_neg = ds.label_counts()
        patients = ds.df["patient_base"].nunique() if "patient_base" in ds.df else ds.df["patient_id"].nunique()
        print(
            f"{split:5s}: events={len(ds.df):3d}, patients={patients:3d}, "
            f"windows={len(ds):5d}, seizure={n_pos:5d}, pre/non={n_neg:5d}"
        )


def main(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"mtgat_soz_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = Path(args.prepared_manifest) if args.prepared_manifest else run_dir / "prepared_manifest.csv"
    df, manifest_summary = prepare_private_soz_manifest(
        args.manifest,
        output_path=prepared_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        infer_from_onset_channels=not args.no_infer_from_onset,
        always_infer_from_onset=args.always_infer_from_onset,
        drop_empty_soz=args.drop_empty_soz,
    )
    print(json.dumps(manifest_summary, indent=2, ensure_ascii=False))
    print(f"Prepared manifest: {prepared_path}")

    if args.dry_run_manifest:
        return

    datasets = build_datasets(df, args)
    if "train" not in datasets:
        raise ValueError("No train split after manifest preparation")
    if "val" not in datasets:
        print("No val split found; using test split for validation metrics.")
        datasets["val"] = datasets.get("test", datasets["train"])
    if "test" not in datasets:
        print("No test split found; using val split for final metrics.")
        datasets["test"] = datasets["val"]

    print_split_info(datasets)
    train_loader = build_loader(datasets["train"], args, shuffle=True)
    val_loader = build_loader(datasets["val"], args, shuffle=False)
    test_loader = build_loader(datasets["test"], args, shuffle=False)

    model = MTGATSOZ(
        n_features=len(TCP_NAMES),
        window_size=args.lookback,
        n_soz=len(TCP_NAMES),
        kernel_size=args.kernel_size,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        use_gatv2=not args.no_gatv2,
        gru_n_layers=args.gru_layers,
        gru_hid_dim=args.gru_hid_dim,
        head_n_layers=args.head_layers,
        head_hid_dim=args.head_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Trainable parameters: {n_params:,}")

    seizure_pos_weight, channel_pos_weight = compute_loss_weights(datasets["train"], device)
    print(f"Seizure pos_weight: {float(seizure_pos_weight.item()):.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.patience // 3),
    )

    best_val = float("inf")
    bad_epochs = 0
    history = []
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            seizure_pos_weight,
            channel_pos_weight,
            args.seizure_loss_weight,
            args.soz_loss_weight,
            args.background_soz_weight,
            args.grad_clip,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            seizure_pos_weight,
            channel_pos_weight,
            args.seizure_loss_weight,
            args.soz_loss_weight,
            args.background_soz_weight,
            neighbour_threshold=args.deepsoz_neighbour_threshold,
        )
        scheduler.step(float(val_metrics["loss"]))
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(make_jsonable(row))
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_det_f1={val_metrics.get('det_f1', float('nan')):.4f} "
            f"val_soz_top1={val_metrics.get('soz_top1_hit', float('nan')):.4f} "
            f"val_corr_sz={val_metrics.get('deepsoz_corr_sz', float('nan')):.4f} "
            f"val_acc_pt={val_metrics.get('deepsoz_acc_pt_weighted', float('nan')):.4f} "
            f"{elapsed:.1f}s"
        )

        if float(val_metrics["loss"]) < best_val:
            best_val = float(val_metrics["loss"])
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "tcp_names": TCP_NAMES,
                    "soz_columns": SOZ_COLUMNS,
                    "best_val_loss": best_val,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        seizure_pos_weight,
        channel_pos_weight,
        args.seizure_loss_weight,
        args.soz_loss_weight,
        args.background_soz_weight,
        neighbour_threshold=args.deepsoz_neighbour_threshold,
        include_deepsoz_details=True,
    )
    test_deepsoz_single = test_metrics.pop("deepsoz_details", None)
    test_deepsoz_mc = None
    if args.mc_samples > 0:
        print(f"\nRunning DeepSOZ MC dropout evaluation (mc_samples={args.mc_samples})...")
        test_deepsoz_mc = compute_deepsoz_metrics_mc(
            model,
            test_loader,
            device,
            mc_samples=args.mc_samples,
            neighbour_threshold=args.deepsoz_neighbour_threshold,
            include_details=True,
        )
    save_deepsoz_results(
        {
            "single_pass": test_deepsoz_single,
            "mc_dropout": test_deepsoz_mc,
        },
        run_dir,
    )

    results = {
        "args": vars(args),
        "manifest_summary": manifest_summary,
        "tcp_names": TCP_NAMES,
        "n_params": n_params,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "test_deepsoz_single_pass": test_deepsoz_single,
        "test_deepsoz_mc_dropout": test_deepsoz_mc,
        "history": history,
        "best_model": str(best_path),
    }
    results_path = run_dir / "results.json"
    results_path.write_text(
        json.dumps(make_jsonable(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTest metrics:")
    print(json.dumps(make_jsonable(test_metrics), indent=2, ensure_ascii=False))
    print(f"Saved: {best_path}")
    print(f"Results: {results_path}")


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Train MTGAT-SOZ on private onset-aligned windows")
    parser.add_argument("--manifest", default="private_manifest_clean.csv")
    parser.add_argument("--data_root", default=r"E:\DataSet\EEG\EEG dataset_SUAT")
    parser.add_argument("--output_dir", default="runs/MTGAT")
    parser.add_argument("--prepared_manifest", default="")
    parser.add_argument("--dry_run_manifest", action="store_true")

    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--pre_sec", type=float, default=15.0)
    parser.add_argument("--post_sec", type=float, default=15.0)
    parser.add_argument("--lookback", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--filter_low", type=float, default=1.0)
    parser.add_argument("--filter_high", type=float, default=50.0)
    parser.add_argument("--cache_size", type=int, default=4)
    parser.add_argument("--max_windows_per_event", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--no_infer_from_onset", action="store_true")
    parser.add_argument("--always_infer_from_onset", action="store_true")
    parser.add_argument("--drop_empty_soz", action="store_true")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    parser.add_argument("--no_gatv2", action="store_true")
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=64)
    parser.add_argument("--head_layers", type=int, default=1)
    parser.add_argument("--head_hid_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.2)

    parser.add_argument("--seizure_loss_weight", type=float, default=1.0)
    parser.add_argument("--soz_loss_weight", type=float, default=1.0)
    parser.add_argument("--background_soz_weight", type=float, default=0.05)
    parser.add_argument("--deepsoz_neighbour_threshold", type=int, default=4)
    parser.add_argument("--mc_samples", type=int, default=20,
                        help="Final DeepSOZ MC dropout samples; set 0 to skip")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
