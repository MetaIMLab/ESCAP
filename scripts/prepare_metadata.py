#!/usr/bin/env python3
"""Stage the released Dataset130_MCWHS metadata into an nnU-Net directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        required=True,
        help="nnUNet_preprocessed root; Dataset130_MCWHS is created below it",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source = repo / "metadata"
    destination = args.preprocessed_root / "Dataset130_MCWHS"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "dataset.json", destination / "dataset.json")
    shutil.copy2(source / "plans.json", destination / "plans.json")
    shutil.copy2(source / "plans.json", destination / "nnUNetPlans.json")
    print(f"Metadata staged in {destination}")


if __name__ == "__main__":
    main()
