#!/usr/bin/env python3
"""Filter combined_manifest.csv to keep only TUSZ rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep rows where source == tusz from a combined manifest CSV."
    )
    parser.add_argument(
        "--input",
        default="combined_manifest.csv",
        help="Path to the source manifest CSV. Default: combined_manifest.csv",
    )
    parser.add_argument(
        "--output",
        default="tusz_manifest.csv",
        help="Path for the filtered manifest CSV. Default: tusz_manifest.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    kept = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is empty or has no header row")
        if "source" not in reader.fieldnames:
            raise ValueError(f"{input_path} does not contain a 'source' column")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                if row.get("source", "").strip().lower() == "tusz":
                    writer.writerow(row)
                    kept += 1
                else:
                    skipped += 1

    print(f"Wrote {kept} TUSZ rows to {output_path}")
    print(f"Skipped {skipped} non-TUSZ rows")


if __name__ == "__main__":
    main()
