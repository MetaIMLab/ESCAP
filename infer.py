#!/usr/bin/env python3
"""Predict source-conditioned ESCAP masks from nnU-Net formatted CT images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-folder", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--preprocessing-workers", type=int, default=2)
    parser.add_argument("--export-workers", type=int, default=2)
    args = parser.parse_args()

    for path in (args.checkpoint, args.input_folder):
        if not path.exists():
            parser.error(f"Path does not exist: {path}")
    if args.preprocessing_workers < 1 or args.export_workers < 1:
        parser.error("Worker counts must be positive")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.nnUNetTrainer.MAPSBalancedThreeEvidenceHeadsTrainer import (
        MAPSBalancedThreeEvidenceHeadsTrainer, source_id,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer_name = MAPSBalancedThreeEvidenceHeadsTrainer.__name__
    if checkpoint.get("trainer_name") != trainer_name:
        parser.error(f"Expected a {trainer_name} checkpoint")
    initialization = checkpoint["init_args"]
    plans_manager = PlansManager(initialization["plans"])
    configuration_manager = plans_manager.get_configuration(initialization["configuration"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    network = MAPSBalancedThreeEvidenceHeadsTrainer.build_network_architecture().to(device).eval()
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=device.type == "cuda", device=device,
        verbose=False, verbose_preprocessing=False, allow_tqdm=True,
    )
    predictor.manual_initialization(
        network, plans_manager, configuration_manager,
        [checkpoint["network_weights"]], initialization["dataset_json"], trainer_name,
        checkpoint.get("inference_allowed_mirroring_axes"),
    )

    images = sorted(args.input_folder.glob("*_0000.nii.gz"))
    if not images:
        parser.error(f"No single-channel nnU-Net CT images in {args.input_folder}")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    for image in images:
        case = image.name.removesuffix("_0000.nii.gz")
        output = args.output_folder / case
        if not args.overwrite and output.with_suffix(".nii.gz").is_file():
            if not args.save_probabilities or output.with_suffix(".npz").is_file():
                print(f"{case}: already predicted", flush=True)
                continue
        protocol = source_id(case)
        predictor.network.set_inference_domain_ids(protocol)
        try:
            predictor.predict_from_files(
                [[str(image)]], [str(output)],
                save_probabilities=args.save_probabilities, overwrite=args.overwrite,
                num_processes_preprocessing=args.preprocessing_workers,
                num_processes_segmentation_export=args.export_workers,
            )
        finally:
            predictor.network.clear_inference_domain_ids()
        print(f"{case}: source {protocol}", flush=True)


if __name__ == "__main__":
    main()
