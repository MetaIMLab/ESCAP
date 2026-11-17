#!/usr/bin/env python3
"""Train the final ESCAP configuration on the three source-mixed folds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


TRAINER = "MAPSBalancedThreeEvidenceHeadsTrainer"
DATASET = "Dataset130_MCWHS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2], choices=[0, 1, 2])
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dataset = args.preprocessed_root / DATASET
    if not args.dry_run:
        for name in ("dataset.json", "nnUNetPlans.json", "splits_final.json"):
            if not (dataset / name).is_file():
                parser.error(f"Required preprocessed file is missing: {dataset / name}")

    environment = os.environ.copy()
    environment.update(
        nnUNet_raw=str(args.raw_root.resolve()),
        nnUNet_preprocessed=str(args.preprocessed_root.resolve()),
        nnUNet_results=str(args.results_root.resolve()),
        nnUNet_compile="False",
    )
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(repo), str(repo / "nnUNet"), environment.get("PYTHONPATH", "")
    )))
    for fold in args.folds:
        command = [sys.executable, "-m", "nnunetv2.run.run_training", "130", "3d_fullres",
                   str(fold), "-tr", TRAINER, "--val_best", "-device", args.device]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repo, env=environment, check=True)


if __name__ == "__main__":
    main()
