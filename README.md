# SynScope Code

## Overview

This is the code repository for SynScope package. It provides tools for image preprocessing, synapse detection, and assignment for multiplex mGRASPi convergence analysis.

## Features

- Image Preprocessing
  - shading correction
  - chromatic shift correction
  - z-signal correction

- Synapse Processing
  - synapse detection
  - synapse assignment

## Installation

### Prerequisite
- Python 3.11
- Conda (recommended) or pip

## Project Structure

```bash

SynScope/
├── synscope_shading_correction.py            # process shading correction to image
├── synscope_chromatic_shift_correction.py    # process chromatic shift coreection
├── synscope_z-signal_correction.py           # process z-signal correction (based on ISCL)
├── synscope_synapse_detection.py             # detect mGRASP puncta
├── synscope_synapse_assignment.py            # assign detected mGRASp puncta   
├── utils/                                    
│   ├── export_puncta_info.py                 # export .nimp metadata to .csv file
│   ├── img_util.py                           # downsample, split, merge images
│   ├── shading_correction.py                 # shading correction core script
│   ├── synapse_classification/              
│   │   ├── mGRASP_puncta_core_functions.py   # puncta assignment core function script
│   │   ├── mGRASP_puncta_feature.py          # puncta assignment feature extraction
│   │   └── mGRASP_puncta_inference.py        # puncta assignment inference using extracted rules
│   └── ISCL/                  
│       ├── models/
│       │   ├── network.py                    # ISCL core network script
│       │   └── trainer.py                    # ISCL core trainer script
│       └── utils/
│           ├── callbacks.py                  # Rollout utilities
│           ├── image_tool.py                 # Rollout utilities
│           ├── metrics.py                    # Rollout utilities
│           ├── normalization.py              # Rollout utilities
│           ├── parser.py                     # Rollout utilities
│           ├── frame_selector.py             # Rollout utilities
│           └── scheduler.py                  # Logging utilities
└── model/        
    ├── _assignment_model/                    # Logging utilities
    ├── _chromatic_shift_parameters/          # Logging utilities
    └── _z_signal_model/                      # Logging utilities

```
### Code Usage
1. **`BC_collapse`**: Processing raw barcode through collapsing cell barcodes.  
2. **`cseq_NucUMIClean`**: Collapsing nucleus barcode information based on UMI.  
3. **`cseq_synaptosome`**: Preprocessing synaptosomal barcode, applying filters to extract cleaned data.  
4. **`cseq_HarmonyIntegration`**: Integrate different connectome-seq datasets to eliminate batch effect.  
5. **`cseq_Annotation`**: Annotate the integrated connectome-seq dataset with Allen Institute mouse brain cell type atlas.  
6. **`cseq_Cluster`**: Cluster the integrated dataset and annotate each cluster from marker gene expression.  
