#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


REGION_NAMES: Tuple[str, ...] = ("FP", "F", "C", "T", "P", "O")


def normalize_region_names(region_names: Sequence[str] | np.ndarray) -> Tuple[str, ...]:
    arr = np.asarray(region_names)
    if arr.ndim != 1:
        raise ValueError(f"region_names must be 1D, got shape={arr.shape}")
    return tuple(str(name) for name in arr.tolist())


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_region_confusion_rows(
    region_probs: np.ndarray,
    region_targets: np.ndarray,
    threshold: float = 0.5,
    region_names: Sequence[str] = REGION_NAMES,
) -> List[dict]:
    region_names = normalize_region_names(region_names)
    probs = np.asarray(region_probs, dtype=np.float32)
    targets = np.asarray(region_targets, dtype=np.float32)
    if probs.shape != targets.shape:
        raise ValueError(
            f"region_probs and region_targets must share the same shape, got "
            f"{probs.shape} vs {targets.shape}"
        )
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got ndim={probs.ndim}")
    if probs.shape[1] != len(region_names):
        raise ValueError(
            f"Expected {len(region_names)} region columns, got {probs.shape[1]}"
        )

    preds = (probs >= threshold).astype(np.int64)
    gold = (targets >= 0.5).astype(np.int64)

    rows: List[dict] = []
    for idx, name in enumerate(region_names):
        pred_col = preds[:, idx]
        gold_col = gold[:, idx]
        tp = int(np.logical_and(pred_col == 1, gold_col == 1).sum())
        fp = int(np.logical_and(pred_col == 1, gold_col == 0).sum())
        tn = int(np.logical_and(pred_col == 0, gold_col == 0).sum())
        fn = int(np.logical_and(pred_col == 0, gold_col == 1).sum())
        support = int(gold_col.sum())
        rows.append(
            {
                "region": name,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "support": support,
                "precision": _safe_div(tp, tp + fp),
                "recall": _safe_div(tp, tp + fn),
                "specificity": _safe_div(tn, tn + fp),
                "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
            }
        )
    return rows


def format_region_confusion_markdown(
    rows: Iterable[dict],
    threshold: float,
) -> str:
    lines = [
        "# Region Confusion Matrix",
        "",
        "The region task is multi-label, so each region is reported as its own binary 2x2 confusion matrix.",
        "",
        f"- Threshold: `{threshold:.3f}`",
        "",
        "## Summary",
        "",
        "| Region | TP | FP | TN | FN | Support | Precision | Recall | Specificity | F1 |",
        "|--------|----|----|----|----|---------|-----------|--------|-------------|----|",
    ]
    for row in rows:
        lines.append(
            f"| {row['region']} | {row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} | "
            f"{row['support']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['specificity']:.4f} | {row['f1']:.4f} |"
        )

    for row in rows:
        lines.extend(
            [
                "",
                f"## {row['region']}",
                "",
                "| actual \\\\ predicted | Negative | Positive |",
                "|----------------------|----------|----------|",
                f"| Negative | {row['tn']} | {row['fp']} |",
                f"| Positive | {row['fn']} | {row['tp']} |",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def save_region_confusion_report(
    region_probs: np.ndarray,
    region_targets: np.ndarray,
    output_dir: Path | str,
    threshold: float = 0.5,
    region_names: Sequence[str] = REGION_NAMES,
) -> Tuple[Path, Path]:
    region_names = normalize_region_names(region_names)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = compute_region_confusion_rows(
        region_probs=region_probs,
        region_targets=region_targets,
        threshold=threshold,
        region_names=region_names,
    )

    md_path = out_dir / "region_confusion_matrix.md"
    csv_path = out_dir / "region_confusion_matrix.csv"

    md_path.write_text(
        format_region_confusion_markdown(rows, threshold=threshold),
        encoding="utf-8",
    )

    csv_lines = [
        "region,tp,fp,tn,fn,support,precision,recall,specificity,f1",
    ]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    row["region"],
                    str(row["tp"]),
                    str(row["fp"]),
                    str(row["tn"]),
                    str(row["fn"]),
                    str(row["support"]),
                    f"{row['precision']:.6f}",
                    f"{row['recall']:.6f}",
                    f"{row['specificity']:.6f}",
                    f"{row['f1']:.6f}",
                ]
            )
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-region binary confusion matrices from test_predictions.npz"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to test_predictions.npz containing region_probs and region_targets",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where region_confusion_matrix.md/csv will be written",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to region probabilities",
    )
    args = parser.parse_args()

    data = np.load(args.predictions)
    if "region_probs" not in data or "region_targets" not in data:
        raise KeyError(
            f"{args.predictions} must contain region_probs and region_targets arrays"
        )
    region_names = REGION_NAMES
    if "region_names" in data:
        region_names = normalize_region_names(data["region_names"])
    md_path, csv_path = save_region_confusion_report(
        region_probs=data["region_probs"],
        region_targets=data["region_targets"],
        output_dir=args.output_dir,
        threshold=args.threshold,
        region_names=region_names,
    )
    print(f"Saved markdown report to {md_path}")
    print(f"Saved csv report to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
