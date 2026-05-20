# SynScope

## Overview

SynScope is a Python toolkit for multiplex **mGRASPi** synaptic convergence. It covers preprocessing (shading, chromatic shift, and z-signal correction), mGRASP puncta detection, and puncta classification for convergence analysis.

## Quick start

```bash
git clone <rhttps://github.com/ychoi68/SynScope.git>
cd SynScope_github

conda create -n synscope python=3.11 -y
conda activate synscope
conda config --append channels conda-forge

# See "Installation" below for the full conda + pip setup
```

Run all scripts from the **repository root** so paths to `model/` resolve correctly.

| Step | Command / script | Typical output |
|------|------------------|----------------|
| 1. Shading | `python synscope_shading_correction.py` | `*_shading_corrected.tiff` |
| 2. Chromatic | `python synscope_chromatic_shift_correction.py` | `*_chromatic_corrected.tiff` |
| 3. Z-signal (optional) | see [Z-signal correction](#3-z-signal-correction) | `*_denoised_image.tiff` |
| 4. Detection | `python synscope_synapse_detection.py` | `*_detected_puncta.nimp` |
| 5. Classification | `python synscope_synapse_classification.py` | `*_predictions.csv`, grouped `.nimp` |

## Features

- **Image preprocessing**
  - Shading correction (BaSiC flatfield on tiled CZI data)
  - Chromatic shift correction (ANTs transforms for LSM780 / LSM980 confocal images)
  - Z-signal correction (ISCL-based denoising on multi-frame TIFF stacks)

- **Synapse processing**
  - mGRASP puncta detection
  - Puncta classification / assignment

## Requirements

- Python 3.11
- [Conda](https://docs.conda.io/) (primary environment manager)

### Installation

Use **conda** for the scientific core and **`zimg`**, then **pip** for everything else. 

**1. Conda (main)**

```bash
conda create -n synscope python=3.11 -y
conda activate synscope
conda config --append channels conda-forge

conda install -y \
  mkl numpy tbb scikit-learn scipy h5py cython ipykernel imageio protobuf future mock \
  shapely pandas seaborn joblib anaconda-client conda-build ninja qt markdown \
  scikit-image matplotlib mkl-service mkl_fft mkl_random

conda install -y zimg -c fenglab
```

**2. Pip (remaining dependencies)**

```bash
pip install --upgrade --no-cache-dir \
  opencv-python yacs anytree termcolor tabulate grpcio tensorboard \
  catboost lightgbm natsort lap pycocotools itk itk-elastix antspyx tensorstore \
  tifffile Pillow tqdm networkx tensorflow tensorflow-addons
```

| Package | Source | Used for |
|---------|--------|----------|
| `numpy`, `scipy`, `pandas`, `scikit-learn`, `scikit-image`, `matplotlib` | conda | Arrays, stats, plotting, classification |
| `zimg` | conda (`fenglab`) | CZI / TIFF / `.nimp` I/O, puncta detection |
| `opencv-python`, `tifffile`, `Pillow`, `tqdm` | pip | Image I/O, z-signal preprocessing |
| `tensorflow`, `tensorflow-addons` | pip | Z-signal (ISCL) inference |
| `antspyx`, `itk`, `itk-elastix` | pip | Chromatic shift registration |
| `networkx`, `catboost`, `lightgbm` | pip | Classification graphs and models |
| `tensorstore` | pip | Tensor I/O (lab stack) |

### Bundled models and parameters

Pretrained assets live under `model/`:

| Path | Purpose |
|------|---------|
| `model/_chromatic_shift_parameters/` | LSM780 / LSM980 chromatic shift transforms |
| `model/_z_signal_model/` | ISCL weights (`my_model_F`, `my_model_H`) |
| `model/_assignment_model/` | Puncta classifier (`model.pkl`, `feature_names.json`) |

## Project structure

```text
SynScope_github/
├── synscope_shading_correction.py         # Shading correction entry point
├── synscope_chromatic_shift_correction.py # Chromatic shift correction
├── synscope_z-signal_correction.py        # Z-signal correction (ISCL)
├── synscope_synapse_detection.py          # mGRASP puncta detection
├── synscope_synapse_classification.py     # Puncta classification
├── synscope_img_util.py                   # Image downsample / merge helpers
├── utils/
│   ├── export_puncta_info.py              # Export .nimp metadata to CSV
│   ├── img_util.py                        # Read / write image utilities
│   ├── shading_correction.py              # BaSiC shading core
│   ├── synpase_classification/            # Puncta feature + inference code
│   │   ├── mGRASP_puncta_core_functions.py
│   │   ├── mGRASP_puncta_feature.py
│   │   └── mGRASP_puncta_inference.py
│   └── ISCL/                              # ISCL network (z-signal)
│       ├── models/
│       └── utils/
└── model/
    ├── _assignment_model/
    ├── _chromatic_shift_parameters/
    └── _z_signal_model/
```

## Typical workflow

1. **Shading correction** on raw CZI → `*_shading_corrected.tiff`
2. **Chromatic shift correction** on the shading-corrected stack
3. **Z-signal correction** (optional) on multi-z TIFF stacks
4. **Puncta detection** → `*_detected_puncta.nimp`
5. **Puncta classification** → CSV predictions and grouped `.nimp` files

---

## Usage

Each top-level script exposes a `main` block you can edit, or you can import and call the functions from your own driver script.

### 1. Shading correction

Corrects illumination inhomogeneity using BaSiC. Input is a **CZI** file; output is a multi-channel TIFF in the same folder (or `result_folder`).

```python
from synscope_shading_correction import shading_correction_convergence

shading_correction_convergence(
    img_file="path/to/sample.czi",
    result_folder=None,              # default: same directory as input
    channels_to_correct=None,        # None = all channels; or e.g. [1, 2]
)
```

```bash
python synscope_shading_correction.py
```

### 2. Chromatic shift correction

Registers a moving channel to channel 1 (fixed) using stored LSM780/LSM980 transforms or a newly computed affine transform.

```python
from synscope_chromatic_shift_correction import apply_chromatic_correction

apply_chromatic_correction(
    img_path="path/to/sample_shading_corrected.tiff",
    scope="lsm980",                  # "lsm980", "lsm780", or "calculate"
    moving_channel=4,                # 1–5 (1-based)
    dtype="auto",                    # "auto", "uint8", or "uint16"
)
```

```bash
python synscope_chromatic_shift_correction.py
```

### 3. Z-signal correction

Denoises z-related signal variation in a **multi-frame TIFF** using ISCL (Lee et al., IEEE TMI 2021).

**Model layout:** inference loads weights from `{result_dir}/model/my_model_F` and `my_model_H`. The bundled checkpoint lives under `model/_z_signal_model/` with those filenames, so copy or link them before running:

```bash
mkdir -p ./z_signal_run/model
cp model/_z_signal_model/my_model_* ./z_signal_run/model/

python synscope_z-signal_correction.py \
  --data path/to/stack.tif \
  --result_dir ./z_signal_run \
  --clean_slide 0 1 2 \
  --noisy_slide 3 4 5 \
  --target_range 10 50 \
  --ref_slide 3 \
  --training false
```

Use `--training true` only when fitting a new model (weights are saved under `{result_dir}/model/`).

| Argument | Description |
|----------|-------------|
| `--data` | Input multi-frame TIFF |
| `--result_dir` | Output directory (writes `*_denoised_image.tiff`; must contain `model/my_model_*` for inference) |
| `--clean_slide` / `--noisy_slide` | Frame indices for ISCL training pairs |
| `--target_range` | Z range to enhance with CLAHE + histogram matching |
| `--ref_slide` | Reference frame for histogram matching |
| `--training` | `true` to train, `false` to run inference (use lowercase strings) |


### 4. Synapse (puncta) detection

Runs mGRASP puncta detection (Feng et al., Bioinformatics 2012). Requires a multi-channel image and voxel sizes in µm.

```python
from synscope_synapse_detection import run_puncta_detection

run_puncta_detection(
    image_folder="path/to/images",
    filename="sample.tiff",
    mGRASP_channel=4,
    dendrite_channel=2,
    threshold=-1,                  # -1 = automatic
    voxelSize_X=0.23,
    voxelSize_Y=0.23,
    voxelSize_Z=0.5,
    swc_name=None,                 # optional dendrite SWC
)
```

Outputs include `*_detected_puncta.nimp`, `*_detected_soma_puncta.nimp`, and logs under `log/`.

```bash
python synscope_synapse_detection.py
```

### 5. Synapse classification

Assigns detected puncta to convergence groups using the model in `model/_assignment_model/` (auto-detected when `model_dir` is omitted).

```python
from synscope_synapse_classification import classify_puncta

results = classify_puncta(
    img_folder="path/to/images",
    img_name="sample.tiff",
    output_dir="path/to/output",
    overlap_thresh=0.6,
    post_cell_channel=2,
    mgrasp_channel=4,
    use_axon_dendrite=True,
)
```

Writes `{stem}_predictions.csv` and per-group `{stem}_{group}.nimp` files under the output directory.

```bash
python synscope_synapse_classification.py
```

### 6. Export puncta metadata

Converts `.nimp` files in a folder to CSV summaries (coordinates, intensity, volume, etc.).

```python
from utils.export_puncta_info import export_puncta_info

export_puncta_info("path/to/puncta/folder")
```

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `model/my_model_F` not found (z-signal) | Copy `model/_z_signal_model/my_model_*` into `{result_dir}/model/` (see above). |
| Classification model not found | Ensure `model/_assignment_model/model.pkl` and `feature_names.json` exist; run from repo root. |
| `zimg` import error | Install `zimg` from your lab distribution; it is not on PyPI. |
| Chromatic shift fails | Confirm `scope` is `lsm980`, `lsm780`, or `calculate`; files under `model/_chromatic_shift_parameters/` must be present. |
| Z-signal uses wrong mode | Pass `--training false` (lowercase). Do not use `False` with capital F unless using the Python API directly. |

---

## References

- **Puncta detection:** Feng et al., *Improved synapse detection for mGRASP-assisted brain connectivity mapping*, Bioinformatics (2012).
- **Z-signal (ISCL):** Lee et al., *ISCL: Interdependent Self-Cooperative Learning for Unpaired Image Denoising*, IEEE TMI (2021).
