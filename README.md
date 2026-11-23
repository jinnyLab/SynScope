# SynScope

## Overview

SynScope is a Python toolkit for multiplex **mGRASPi** synaptic convergence analysis. It provides an integrated pipeline for image preprocessing, mGRASP puncta detection, and puncta classification for presynaptic input convergence analysis.

The pipeline includes:

- Shading correction using BaSiC
- Chromatic shift correction using ANTs
- Z-signal correction using ISCL
- mGRASP puncta detection
- Synapse classification and assignment

**Source code:** https://github.com/jinnyLab/SynScope

**Online demo:** https://synscope-mgrasp.com  
**Access code:** `synscope`

## Quick start

```bash
git clone https://github.com/jinnyLab/SynScope.git
cd SynScope
```

See the Installation section below for environment setup.

Run all scripts from the **repository root** so that paths to `model/` and `utils/` resolve correctly.

| Step | Script / operation | Typical output |
|---|---|---|
| 1. Shading correction | `synscope_shading_correction.py` | `*_shading_corrected.tiff` |
| 2. Chromatic shift correction | `synscope_chromatic_shift_correction.py` | `*_chromatic_corrected.tiff` |
| 3. Z-signal correction | Split → correction → merge | `*_denoised_image.tiff`, `*_merged.tiff` |
| 4. Puncta detection | `synscope_synapse_detection.py` | `*_detected_puncta.nimp` |
| 5. Classification | `synscope_synapse_classification.py` | `*_predictions.csv`, grouped `.nimp` files |

## System requirements

### Tested environment

SynScope was tested using the following software environment:

| Component | Tested version |
|---|---|
| Operating system | Ubuntu 20.04.2 LTS |
| Python | 3.10.13 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| pandas | 2.2.1 |
| scikit-learn | 1.3.0 |
| scikit-image | 0.22.0 |
| TensorFlow | 2.13.0 |
| LightGBM | 3.3.5 |
| ANTsPy | 0.3.8 |
| OpenCV | 4.10.0 |
| ITK | 5.3.0 |
| ITK-Elastix | 0.17.1 |
| NetworkX | 3.1 |
| zimg | 2023.07.13.10 |

**Hardware:** The complete pipeline was tested on an NVIDIA DGX Station equipped with four Tesla V100-DGXS GPUs (16 GB each).

A GPU is recommended for ISCL-based z-signal correction. CPU execution is possible but may take considerably longer, particularly for large image stacks.

## Installation

### 1. Create the Conda environment

Install Conda or Miniconda if it is not already available.

Create and activate a new environment:

```bash
conda create -n synscope python=3.10.13 -y
conda activate synscope
```

### 2. Install dependencies

Install the scientific computing packages:

```bash
conda install -y -c conda-forge \
    numpy=1.26.4 \
    scipy=1.11.4 \
    pandas=2.2.1 \
    scikit-learn=1.3.0 \
    scikit-image=0.22.0 \
    networkx=3.1 \
    h5py \
    joblib
```

Install `zimg` from the FengLab Conda channel:

```bash
conda install -y -c fenglab zimg
```

Install the remaining Python dependencies:

```bash
pip install \
    opencv-python==4.10.0 \
    tifffile \
    Pillow \
    tqdm \
    lightgbm==3.3.5 \
    tabulate \
    termcolor \
    antspyx==0.3.8 \
    tensorflow==2.13.0 \
    tensorflow-addons==0.23.0 \
    itk==5.3.0 \
    itk-elastix==0.17.1
```

### 3. Verify the installation

Run the following commands to check that the main dependencies are available:

```python
import numpy
import scipy
import pandas
import sklearn
import skimage
import tensorflow
import lightgbm
import ants
import cv2
import zimg

print("SynScope dependencies imported successfully.")
```

### Installation time

Installation takes approximately **1 hour on a standard desktop computer**. Actual installation time may vary depending on internet connection speed, hardware, and package dependency resolution.

## Demo dataset and instructions

A sample microscopy dataset for demonstrating the complete SynScope pipeline is provided separately from the GitHub repository.

The demo dataset includes instructions and reference outputs to help users verify the results at each processing stage.

### Demo workflow and expected outputs

The demo covers the complete SynScope workflow, from image preprocessing to mGRASP puncta detection and classification.

| Processing step | Description | Expected output |
|---|---|---|
| 1. Shading correction | Corrects illumination inhomogeneity in raw microscopy images using BaSiC. | `*_shading_corrected.tiff` |
| 2. Chromatic shift correction | Aligns fluorescence channels to correct chromatic displacement. | `*_chromatic_corrected.tiff` |
| 3. Z-signal correction | Corrects signal attenuation across z-planes using ISCL. | `*_denoised_image.tiff` |
| 4. Channel merging | Recombines individually processed channels into a multichannel image. | `*_merged.tiff` |
| 5. mGRASP puncta detection | Detects mGRASP-positive synaptic puncta. | `*_detected_puncta.nimp` and `*_detected_soma_puncta.nimp` |
| 6. Puncta classification | Classifies detected puncta according to presynaptic input identity. | `*_predictions.csv` and grouped `*.nimp` files |

The main output files include:

- **Preprocessed microscopy images:** Multichannel TIFF files after shading, chromatic shift, and z-signal correction.
- **Detected puncta:** `.nimp` files containing detected mGRASP puncta.
- **Classification results:** CSV files containing puncta classification predictions.
- **Grouped puncta:** `.nimp` files containing puncta assigned to individual presynaptic input groups.

### Expected runtime

The complete demo workflow typically takes approximately **1–2 hours on a standard desktop computer**. Runtime depends on image dimensions, the number of channels and z-planes, and available hardware.

## Bundled models and parameters

Pretrained models and processing parameters are included under `model/`.

| Path | Purpose |
|---|---|
| `model/_chromatic_shift_parameters/` | LSM780 / LSM980 chromatic shift transforms |
| `model/_z_signal_model/` | ISCL model weights (`my_model_F`, `my_model_H`) |
| `model/_assignment_model/` | Puncta classifier (`model.pkl`, `feature_names.json`) |

## Project structure

```text
SynScope/
├── synscope_shading_correction.py
├── synscope_chromatic_shift_correction.py
├── synscope_z-signal_correction.py
├── synscope_synapse_detection.py
├── synscope_synapse_classification.py
├── synscope_img_util.py
├── utils/
│   ├── export_puncta_info.py
│   ├── img_util.py
│   ├── shading_correction.py
│   ├── synpase_classification/
│   │   ├── mGRASP_puncta_core_functions.py
│   │   ├── mGRASP_puncta_feature.py
│   │   └── mGRASP_puncta_inference.py
│   └── ISCL/
│       ├── models/
│       └── utils/
└── model/
    ├── _assignment_model/
    ├── _chromatic_shift_parameters/
    └── _z_signal_model/
```

## Typical workflow

1. **Shading correction** on raw CZI images.
2. **Chromatic shift correction** on the shading-corrected images.
3. **Channel splitting** before z-signal correction.
4. **Z-signal correction** on each single-channel TIFF stack.
5. **Channel merging** to reconstruct the multichannel image.
6. **Synapse detection** to identify mGRASP puncta.
7. **Synapse classification** to assign detected puncta to presynaptic input groups.

## Usage

Each top-level script can be run independently or imported into a custom Python workflow. The following examples illustrate the main functions.

### 1. Shading correction

Corrects illumination inhomogeneity using BaSiC. The input is a CZI file.

```python
from synscope_shading_correction import shading_correction_convergence

shading_correction_convergence(
    img_file="path/to/sample.czi",
    result_folder=None,
    channels_to_correct=None,
)
```

### 2. Chromatic shift correction

Registers a moving channel to channel 1 (fixed) using stored LSM780/LSM980 transforms or a newly computed affine transform.

```python
from synscope_chromatic_shift_correction import apply_chromatic_correction

apply_chromatic_correction(
    img_path="path/to/sample_shading_corrected.tiff",
    scope="lsm980",
    moving_channel=4,
    dtype="auto",
)
```

### 3. Z-signal correction

Uses ISCL to denoise z-related signal variation in multi-frame TIFF stacks. Multi-channel images must be split before correction and merged afterward.

**Split channels:**

```python
from synscope_img_util import channel_split

channel_split(
    image_folder="path/to/images",
    filename="sample_chromatic_corrected.tiff",
    result_folder="path/to/images/channel_split",
)
```

**Run z-signal correction:**

The inference model expects the bundled weights under `{result_dir}/model/`.

```bash
mkdir -p ./z_signal_run/model
cp model/_z_signal_model/my_model_* ./z_signal_run/model/

python synscope_z-signal_correction.py \
  --data path/to/images/channel_split/sample_chromatic_corrected_ch4.tiff \
  --result_dir ./z_signal_run/ch4 \
  --clean_slide 0 1 2 \
  --noisy_slide 3 4 5 \
  --target_range 10 50 \
  --ref_slide 3 \
  --training false
```

Repeat for each channel to be corrected.

**Merge channels:**

```python
from synscope_img_util import merge_channel

merge_channel("path/to/denoised_channels_folder")
```

### 4. Synapse (puncta) detection

Detects mGRASP puncta from multichannel microscopy images. Voxel dimensions must be provided in micrometers.

```python
from synscope_synapse_detection import run_puncta_detection

run_puncta_detection(
    image_folder="path/to/images",
    filename="sample.tiff",
    mGRASP_channel=4,
    dendrite_channel=2,
    threshold=-1,
    voxelSize_X=0.23,
    voxelSize_Y=0.23,
    voxelSize_Z=0.5,
    swc_name=None,
)
```

Outputs include detected puncta `.nimp` files and processing logs.

### 5. Synapse classification

Assigns detected puncta to presynaptic convergence groups using the pretrained classifier.

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

The output includes a CSV prediction file and individual `.nimp` files for each classified group.

### 6. Export puncta metadata

Converts `.nimp` files into CSV summaries containing puncta information.

```python
from utils.export_puncta_info import export_puncta_info

export_puncta_info("path/to/puncta/folder")
```

## Troubleshooting

| Issue | What to check |
|---|---|
| `zimg` import error | Install `zimg` from the FengLab Conda channel and activate the correct environment. |
| Z-signal model not found | Copy `my_model_F` and `my_model_H` into `{result_dir}/model/`. |
| Classification model not found | Confirm that `model/_assignment_model/model.pkl` and `feature_names.json` exist. |
| Chromatic shift correction fails | Check the selected scope and confirm that the corresponding transform files are present. |
| Z-signal correction is slow | GPU execution is recommended for large image stacks. |
| Out-of-memory errors | Reduce image stack size or frame count, or use a machine with more available memory. |
| Empty puncta detection output | Check channel assignments, voxel dimensions, and detection threshold. |
| Missing `.nimp` files during classification | Run puncta detection first and confirm that the output files are in the classification input folder. |
| Import or path errors | Run scripts from the repository root. |

## References

- **Puncta detection:** Feng et al., *Improved synapse detection for mGRASP-assisted brain connectivity mapping*, Bioinformatics (2012).
- **Z-signal correction (ISCL):** Lee et al., *ISCL: Interdependent Self-Cooperative Learning for Unpaired Image Denoising*, IEEE Transactions on Medical Imaging (2021).
- **Shading correction:** Peng et al., *A BaSiC tool for background and shading correction of optical microscopy images*, Nature Communications (2017).

## Code and data availability

The SynScope source code is available at https://github.com/jinnyLab/SynScope.

The raw microscopy datasets and processed data supporting the findings of this study are available from the corresponding author upon reasonable request.

## License

SynScope is licensed under the BSD 3-Clause License. See [LICENSE.txt](LICENSE.txt) for the full text.
