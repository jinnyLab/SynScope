#!/usr/bin/env python3
import os
import sys

from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from zimg import ZPuncta
from utils.synpase_classification.mGRASP_puncta_core_functions import (
    ClassifyConfig,
    load_data,
    compute_adaptive_thresholds,
    separate_axon_dendrite,
    process_punctum_channels,
)
from utils.synpase_classification.mGRASP_puncta_inference import PairwiseOverlapClassifier


def _find_models() -> Optional[str]:
    """
    Automatically find unified overlap model in the model folder in the same directory as the script.

    Returns:
        Path to model directory if found, None otherwise
    """
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir / "model/_assignment_model"

    if not model_dir.is_dir():
        return None

    for name in ("model.pkl", "unified_overlap_model.pkl"):
        if (model_dir / name).is_file():
            return str(model_dir)
    return None


# Default threshold for channel 5 pairs
DEFAULT_CH5_OVERLAP_THRESH = 0.5


def classify_puncta(
    img_folder: str,
    img_name: str,
    model_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    overlap_thresh: float = 0.6,
    post_cell_channel: int = 2,
    mgrasp_channel: int = 4,
    use_axon_dendrite: bool = True,
    channel_threshold_multipliers: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Supports images with 3, 4, or 5 channels. The mGRASP channel is always excluded
    from analysis. The post-cell channel may be included or excluded based on
    the use_axon_dendrite parameter.

    Args:
        img_folder: Path to folder containing image and .nimp files
        img_name: Name of the image file
        model_dir: Path to directory containing trained model (optional - will auto-detect from model/ folder if not provided)
        output_dir: Output directory for results (default: {img_folder}/puncta_classification)
        overlap_thresh: Probability threshold for pairwise overlap prediction (default: 0.6)
        post_cell_channel: Channel to use for dendrite subtraction (default: 2)
        mgrasp_channel: mGRASP channel - always excluded from analysis (default: 4)
        use_axon_dendrite: If True, post-cell_channel is included in analysis;
                          if False, it is excluded (default: True)

    Returns:
        Dictionary containing:
            - 'results': List of dicts with predictions for each punctum
            - 'puncta_groups': Dict mapping predictions to lists of puncta objects
            - 'n_predictions': Number of predictions made
            - 'output_dir': Output directory where results were saved
    """

    # Auto-detect model directory if not provided
    if model_dir is None:
        model_dir = _find_models()
        if model_dir is None:
            script_dir = Path(__file__).resolve().parent
            model_dir = script_dir / "model"
            raise ValueError(
                "No model directory provided and auto-detection failed.\n"
                f"Please specify model_dir or ensure models are in: {model_dir}"
            )

    # Verify model directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(img_folder, "puncta_classification")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    print("Step 1: Loading data...")
    img_name_loaded, channel_map, axon_dendrite_image, punctum_list, roi_intensity_samples = load_data(
        img_folder,
        img_name,
        mgrasp_channel=mgrasp_channel,
        post_cell_channel=post_cell_channel if use_axon_dendrite else None,
        use_axon_dendrite=use_axon_dendrite,
    )
    print(f"  Loaded {len(punctum_list)} puncta")
    print()

    # Step 2: Compute adaptive thresholds
    print("Step 2: Computing adaptive thresholds...")
    classify_config = ClassifyConfig()
    adaptive_thresholds = compute_adaptive_thresholds(roi_intensity_samples, classify_config)

    print()

    # Step 3: Separate axon/dendrite if needed
    axon_mask = None
    dendrite_mask = None
    if use_axon_dendrite and axon_dendrite_image is not None and post_cell_channel is not None:
        print("Step 3: Separating axon/dendrite...")
        axon_mask, dendrite_mask = separate_axon_dendrite(
            axon_dendrite_image,
            adaptive_thresh=adaptive_thresholds.get(post_cell_channel, 0.0),
            thickness_thresh=classify_config.thickness_thresh,
            soft_margin=classify_config.soft_margin,
        )
        print()

    # Step 4: Initialize classifier
    print("Step 4: Loading overlap models...")
    try:
        classifier = PairwiseOverlapClassifier(
            model_dir,
            overlap_thresh=overlap_thresh,
            ch5_overlap_thresh=DEFAULT_CH5_OVERLAP_THRESH,
        )
        print("  Models loaded successfully")
        print()
    except Exception as e:
        print(f"  ERROR: Failed to load models: {e}")
        raise

    # Step 5: Classify each punctum
    print("Step 5: Classifying puncta...")
    results = []
    puncta_groups = defaultdict(list)

    excluded_channels = {mgrasp_channel}
    if not use_axon_dendrite and post_cell_channel in channel_map:
        excluded_channels.add(post_cell_channel)

    for i, punctum in enumerate(punctum_list):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(punctum_list)} puncta...")

        # Process channels for this punctum
        z_mask_map, detected_channels = process_punctum_channels(
            punctum,
            channel_map,
            adaptive_thresholds,
            punctum_id=i,
            excluded_channels=excluded_channels,
            axon_mask=axon_mask,
            post_cell_channel=post_cell_channel if use_axon_dendrite else None,
        )

        if not detected_channels:
            # No channels detected
            result = {
                'punctum_id': i,
                'final_prediction': 'low_confidence',
                'final_confidence': 0.0,
                'x': float(punctum.x),
                'y': float(punctum.y),
                'z': float(punctum.z),
            }
            results.append(result)
            puncta_groups['low_confidence'].append(punctum)
            continue

        # Classify using pairwise overlap
        try:
            classification_result = classifier.predict_single_punctum(
                punctum,
                z_mask_map,
                channel_map,
                adaptive_thresholds,
                axon_mask=axon_mask,
                dendrite_mask=dendrite_mask,
                mgrasp_channel=mgrasp_channel,
            )

            final_pred = classification_result.get('prediction', 'low_confidence')
            final_conf = classification_result.get('confidence', 0.0)

            result = {
                'punctum_id': i,
                'final_prediction': final_pred,
                'final_confidence': float(final_conf),
                'x': float(punctum.x),
                'y': float(punctum.y),
                'z': float(punctum.z),
            }
            results.append(result)
            puncta_groups[final_pred].append(punctum)

        except Exception as e:
            print(f"  Warning: Error classifying punctum {i}: {e}")
            result = {
                'punctum_id': i,
                'final_prediction': 'low_confidence',
                'final_confidence': 0.0,
                'x': float(punctum.x),
                'y': float(punctum.y),
                'z': float(punctum.z),
            }
            results.append(result)
            puncta_groups['low_confidence'].append(punctum)

    print(f"  Completed: {len(results)} puncta classified")
    print()

    # Step 6: Save results
    print("Step 6: Saving results...")
    stem = Path(img_name).stem

    # Save CSV
    csv_path = os.path.join(output_dir, f"{stem}_predictions.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Save grouped .nimp files
    for group_name, plist in puncta_groups.items():
        if not plist:
            continue
        nimp_path = os.path.join(output_dir, f"{stem}_{group_name}.nimp")
        try:
            ZPuncta(plist).save(nimp_path)
            print(f"  Saved {group_name}: {len(plist)} puncta -> {nimp_path}")
        except Exception as e:
            print(f"  Warning: Failed to save {group_name}: {e}")


    return {
        'results': results,
        'puncta_groups': dict(puncta_groups),
        'n_predictions': len(results),
        'output_dir': output_dir,
    }


if __name__ == "__main__":

    img_folder = "path/to/image/folder"
    img_name = "image_name.tiff"

    output_dir = "path/to/output_dir"

    # Run classification
    results = classify_puncta(
        img_folder=img_folder,
        img_name=img_name,
        output_dir=output_dir,
        overlap_thresh=0.6,
        post_cell_channel=2,
        mgrasp_channel=4,
        use_axon_dendrite=True,
    )

    print(f"\nResults saved to: {results['output_dir']}")
