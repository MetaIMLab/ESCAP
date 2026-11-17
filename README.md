# ESCAP

**Evidential Source-Conditioned Segmentation for Heterogeneous Medical Image Annotation Protocols**

This repository contains the final MAPS implementation used by ESCAP. MAPS
trains one 3D model across CHD68, ImageALCAPA, and MM-WHS while conditioning
segmentation on the known source protocol. The released checkpoint family uses
three independent source-specific evidence heads and the balanced objective

    L = L_seg + 0.10 L_EDL + 0.10 L_ESA

The repository does not redistribute medical images, labels, or checkpoints.
The three-fold EDL=0.10, ESA=0.10 checkpoints remain in the local release
workspace and are excluded from the public Git repository.

## Contents

    ESCAP/
    |- infer.py                         # nnU-Net sliding-window inference
    |- scripts/train_maps.py            # three-fold training entry point
    |- src/model/MAPS/                  # MAPS network and convolution blocks
    |- nnUNet/                          # nnU-Net core and final MAPS trainer
    |- metadata/                        # Dataset130_MCWHS metadata and plans
    |- weights/edl010_esa010/           # metadata and local-only checkpoints
    |- docs/DATASETS.md                 # dataset descriptions and source mapping
    `- tests/test_maps_model.py         # CPU architecture smoke test

## Environment

The code was developed with Python 3.10, PyTorch 2.x, CUDA, and the nnU-Net
dependencies listed in `requirements.txt`.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    export PYTHONPATH="$PWD:$PWD/nnUNet${PYTHONPATH:+:$PYTHONPATH}"

For a GPU installation, install the PyTorch build matching the local CUDA
runtime before installing the remaining requirements.

## Inference

The input must be an nnU-Net formatted dataset with one CT channel. The case
name must preserve the source prefix used during training:

| Source | Case prefix | Source id |
| --- | --- | --- |
| CHD68 | `CHD68_` | 0 |
| ImageALCAPA | `ImageALCAPA_` | 1 |
| MM-WHS | `MM_WHS_` | 2 |

The input folder should contain `.nii.gz` images named with the `_0000` channel
suffix. The checkpoint contains the plans and label map used for inference.

Place a compatible checkpoint at the path shown below, then run one fold:

    python infer.py \
      --checkpoint weights/edl010_esa010/fold_0/checkpoint_best.pth \
      --input-folder /path/to/imagesTs \
      --output-folder /path/to/predictions

Use `fold_1` or `fold_2` for the other local checkpoints. Source identity
is inferred from each case name and selected before sliding-window prediction.
The auxiliary evidence output is not used as the final mask.

## Training

Training requires a preprocessed MC-WHS collection and a fixed three-fold
split. The medical data and split files are intentionally not part of this
repository. Run the final configuration on each fold with:

    python scripts/train_maps.py \
      --raw-root /path/to/nnUNet_raw \
      --preprocessed-root /path/to/nnUNet_preprocessed \
      --results-root /path/to/nnUNet_results

This entry point trains from random initialization with batch size 8, 100
epochs, AdamW, a base learning rate of `1e-4`, a separate `1e-3` evidence-head
learning rate, and source ids CHD68=0, ImageALCAPA=1, MM-WHS=2.

## Data

See [docs/DATASETS.md](docs/DATASETS.md) for the CHD68, ImageALCAPA, and
MM-WHS descriptions. Users must obtain the datasets from their respective
sources and follow their licenses.

## License and third-party code

ESCAP/MAPS source code has no separate license declaration in the source
workspace. See [NOTICE.md](NOTICE.md) before redistribution. The bundled
nnU-Net source is retained to make the trainer reconstructible; its upstream
Apache-2.0 license is in `nnUNet/LICENSE`.
