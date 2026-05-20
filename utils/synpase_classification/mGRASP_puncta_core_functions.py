#!/usr/bin/env python3
"""
Core functions for mGRASP puncta classification.

This module contains essential functions for data loading, thresholding, and axon/dendrite separation
used by the ML-based classification pipeline.
"""

import os
import sys

from pathlib import Path,PurePath
sys.path.append('../')

from zimg import *

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_dilation as binary_dilation_3d
from skimage.measure import label, regionprops


# =========================
# Configuration Classes
# =========================

@dataclass
class ClassifyConfig:
    """Configuration for puncta classification thresholding and preprocessing."""
    # Thresholding
    fallback_percentile: int = 95
    gmm_alpha: float = 1.0
    gmm_dprime_thresh: float = 1.0

    # Axon/dendrite separation
    thickness_thresh: float = 6.0
    soft_margin: float = 0.5

# =========================
# Thresholding Functions
# =========================

def fit_threshold_gmm(
    pixels: np.ndarray,
    fallback_percentile: int = 95,
    dprime_thresh: float = 1.0,
    return_dprime: bool = False,
) -> Tuple[float, Optional[float]]:
    """Robust threshold via 2-comp GMM with d' check; percentile fallback."""
    pixels = np.asarray(pixels).reshape(-1, 1)
    if len(pixels) < 10:
        thr = float(np.percentile(pixels, fallback_percentile))
        return (thr, None) if return_dprime else (thr,)

    try:
        gmm = GaussianMixture(n_components=2, random_state=0).fit(pixels)
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        order = np.argsort(means)
        means, stds = means[order], stds[order]
        dprime = abs(means[1] - means[0]) / np.sqrt(0.5 * (stds[0] ** 2 + stds[1] ** 2))

        if dprime < dprime_thresh:
            thr = float(np.percentile(pixels, fallback_percentile))
        else:
            # Use midpoint threshold (50% towards signal mean)
            thr = float(means[0] + 0.5 * (means[1] - means[0]))
        return (thr, float(dprime)) if return_dprime else (thr,)

    except Exception:
        # Fallback to percentile if GMM fails
        thr = float(np.percentile(pixels, fallback_percentile))
        return (thr, None) if return_dprime else (thr,)

def compute_adaptive_thresholds(roi_intensity_samples: Dict[int, List[float]], cfg: ClassifyConfig) -> Dict[int, float]:
    """Global thresholds per channel using GMM + d′ with safe fallbacks."""
    adaptive_thresholds: Dict[int, float] = {}

    for ch_num, pixels in roi_intensity_samples.items():
        pixels = np.array(pixels)
        if len(pixels) < 10:
            adaptive_thresholds[ch_num] = float(np.percentile(pixels, cfg.fallback_percentile)) if len(pixels) else 0.0
            continue

        thr, dprime = fit_threshold_gmm(
            pixels,
            fallback_percentile=cfg.fallback_percentile,
            dprime_thresh=cfg.gmm_dprime_thresh,
            return_dprime=True,
        )

        adaptive_thresholds[ch_num] = float(thr)

    return adaptive_thresholds


# =========================
# Axon / Dendrite Functions
# =========================

def separate_axon_dendrite(ch2_image: np.ndarray, adaptive_thresh: float, thickness_thresh: float = 6.0, soft_margin: float = 1.0):
    """Separate axon and dendrite based on thickness morphology."""
    axon_mask = np.zeros_like(ch2_image, dtype=bool)
    dendrite_mask = np.zeros_like(ch2_image, dtype=bool)
    all_props = []

    for z in range(ch2_image.shape[0]):
        plane = ch2_image[z]
        if np.count_nonzero(plane) == 0:
            continue
        binary = plane > adaptive_thresh
        labeled = label(binary)
        props = regionprops(labeled)
        for prop in props:
            area = prop.area
            minr, minc, maxr, maxc = prop.bbox
            length = max(maxr - minr, maxc - minc)
            thickness = area / (length + 1e-5)
            coords = (z, prop.coords[:, 0], prop.coords[:, 1])
            all_props.append((coords, thickness))

    for coords, thickness in all_props:
        if thickness >= thickness_thresh + soft_margin:
            dendrite_mask[coords] = 1
        elif thickness <= thickness_thresh - soft_margin:
            axon_mask[coords] = 1
        else:
            dendrite_mask[coords] = 1
            axon_mask[coords] = 1

    axon_mask = binary_dilation_3d(axon_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
    dendrite_mask = dendrite_mask.astype(np.uint8)
    return axon_mask.astype(np.uint8), dendrite_mask.astype(np.uint8)

# =========================
# Data Loading Functions
# =========================

def load_data(
    img_folder: str,
    img_name: str,
    mgrasp_channel: int,
    axon_dendrite_channel: Optional[int] = None,
    use_axon_dendrite: bool = True
):
    """Load image data and puncta from folder.

    Args:
        img_folder: Path to folder containing image and .nimp file
        img_name: Name of the image file
        mgrasp_channel: Channel containing mGRASP signal - always excluded from analysis
        axon_dendrite_channel: Channel to use for axon/dendrite morphology analysis (optional)
        use_axon_dendrite: If True, axon_dendrite_channel is included in analysis; if False, excluded

    Returns:
        Tuple of (img_name, channel_map, axon_dendrite_image, punctum_list, roi_intensity_samples)
        - channel_map: Dictionary of channels to use for overlap analysis (excludes mgrasp_channel,
                       and axon_dendrite_channel if use_axon_dendrite=False)
        - axon_dendrite_image: Image for axon/dendrite separation (None if not provided/used)
    """
    img_path = os.path.join(img_folder, img_name)
    img_infos = ZImg.readImgInfos(img_path)
    num_image_planes = img_infos[0].numChannels

    # Validate channel numbers
    if mgrasp_channel < 1 or mgrasp_channel > num_image_planes:
        raise ValueError(f"mGRASP channel {mgrasp_channel} is out of range. Image has {num_image_planes} channels.")

    if axon_dendrite_channel is not None:
        if axon_dendrite_channel < 1 or axon_dendrite_channel > num_image_planes:
            raise ValueError(f"Axon/dendrite channel {axon_dendrite_channel} is out of range. Image has {num_image_planes} channels.")
        if axon_dendrite_channel == mgrasp_channel:
            raise ValueError(f"Axon/dendrite channel cannot be the same as mGRASP channel ({mgrasp_channel}).")

    imgObj = ZImg(img_path, scene=0, xRatio=1, yRatio=1)
    img = imgObj.data[0]
    if img.max() > 255:
        img = (img / img.max()) * 255.0

    # Build channel_map for all available channels (1-indexed)
    all_channels = {}
    for i in range(num_image_planes):
        all_channels[i + 1] = img[i]

    # Always exclude mGRASP channel from analysis
    channel_map = {ch: img_data for ch, img_data in all_channels.items() if ch != mgrasp_channel}

    # Get axon/dendrite image if provided
    axon_dendrite_image = None
    if axon_dendrite_channel is not None:
        axon_dendrite_image = all_channels[axon_dendrite_channel]

    # Exclude axon_dendrite_channel from analysis if not used
    if not use_axon_dendrite and axon_dendrite_channel is not None:
        if axon_dendrite_channel in channel_map:
            del channel_map[axon_dendrite_channel]

    # Validate that we have at least one channel for analysis
    if not channel_map:
        raise ValueError(
            f"No channels available for analysis. "
            f"mGRASP channel {mgrasp_channel} is excluded, "
            f"and axon/dendrite channel {axon_dendrite_channel} is also excluded."
        )

    punctum_list = []
    image_stem = os.path.splitext(img_name)[0]
    matched_nimp_files = []
    fallback_nimp_files = []

    for fn in sorted(os.listdir(img_folder)):
        if not fn.endswith(".nimp"):
            continue

        fn_lower = fn.lower()
        # Exclude filtered puncta outputs from downstream classification input.
        if (
            "_filtered_puncta.nimp" in fn_lower
            or "_filtered_soma_puncta.nimp" in fn_lower
            or "puncta_filtered_puncta" in fn_lower
            or "detected_soma_puncta_filtered_soma_puncta" in fn_lower
        ):
            continue

        # Accept multiple puncta file variants:
        # - "*_detected_puncta.nimp"
        # - "*_detected_soma_puncta.nimp"
        # - "*_puncta.nimp"
        # - "*_soma_puncta.nimp"
        # - assignment variants containing "assign" (e.g., "*assignment*.nimp")
        is_supported_nimp = (
            fn.endswith("_detected_puncta.nimp")
            or fn.endswith("_detected_soma_puncta.nimp")
            or fn.endswith("_puncta.nimp")
            or fn.endswith("_soma_puncta.nimp")
            or "assign" in fn_lower
        )
        if not is_supported_nimp:
            continue

        # Prefer files corresponding to the current image stem, but support fallback.
        if image_stem in fn:
            matched_nimp_files.append(fn)
        else:
            fallback_nimp_files.append(fn)

    nimp_files_to_load = matched_nimp_files if matched_nimp_files else fallback_nimp_files
    for fn in nimp_files_to_load:
        puncta = ZPuncta(os.path.join(img_folder, fn))
        punctum_list.extend(puncta.data)

    if nimp_files_to_load:
        print(f"[INFO] Loaded puncta from {len(nimp_files_to_load)} .nimp file(s): {nimp_files_to_load}")

    roi_intensity_samples = {ch: [] for ch in channel_map}
    x_size, y_size = 5, 5

    # Sample ROIs from puncta locations
    if len(punctum_list) > 0:
        for punctum in tqdm(punctum_list, desc="Sampling intensities from puncta"):
            x_max, x_min = int(np.max(punctum.voxelLocations[:, 0])), int(np.min(punctum.voxelLocations[:, 0]))
            y_max, y_min = int(np.max(punctum.voxelLocations[:, 1])), int(np.min(punctum.voxelLocations[:, 1]))
            slice_z = int(punctum.z)
            first_ch_img = next(iter(channel_map.values()))
            max_z = first_ch_img.shape[0] - 1

            if slice_z < 0:
                slice_z_clamped = 0
            elif slice_z > max_z:
                slice_z_clamped = max_z
            else:
                slice_z_clamped = slice_z

            for ch_num, ch_img in channel_map.items():
                h, w = ch_img.shape[1:]
                ch_max_z = ch_img.shape[0] - 1
                if slice_z_clamped > ch_max_z:
                    slice_z_clamped_ch = ch_max_z
                elif slice_z_clamped < 0:
                    slice_z_clamped_ch = 0
                else:
                    slice_z_clamped_ch = slice_z_clamped

                x1, x2 = max(0, x_min - x_size), min(w, x_max + x_size)
                y1, y2 = max(0, y_min - y_size), min(h, y_max + y_size)
                roi = ch_img[slice_z_clamped_ch, y1:y2, x1:x2]
                if roi.size > 0:
                    roi_intensity_samples[ch_num].extend(roi.flatten())

    return img_name, channel_map, axon_dendrite_image, punctum_list, roi_intensity_samples


# =========================
# Utility Functions
# =========================

def process_punctum_channels(
    punctum,
    channel_map: Dict[int, np.ndarray],
    adaptive_thresholds: Dict[int, float],
    punctum_id: int,
    excluded_channels: set = None,
    axon_mask: np.ndarray = None,
    axon_dendrite_channel: int = None
) -> Tuple[Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]], List[int]]:
    """
    Process channels for a single punctum and extract masks.

    Args:
        punctum: Punctum object
        channel_map: Channel images
        adaptive_thresholds: Pre-computed adaptive thresholds
        punctum_id: ID of the punctum for error reporting
        excluded_channels: Set of channel numbers to exclude from analysis
        axon_mask: 3D axon mask (Z, Y, X) - used to filter channel 2 masks
        axon_dendrite_channel: Channel number for axon/dendrite separation (typically 2)

    Returns:
        Tuple of (z_mask_map, detected_channels)
    """
    try:
        if not hasattr(punctum, 'voxelLocations') or punctum.voxelLocations is None:
            return {}, []

        if punctum.voxelLocations.shape[0] < 5:
            return {}, []

        voxel_locs = np.array(punctum.voxelLocations)
        if voxel_locs.ndim != 2 or voxel_locs.shape[1] != 3:
            return {}, []

        x_max, x_min = int(np.max(voxel_locs[:, 0])), int(np.min(voxel_locs[:, 0]))
        y_max, y_min = int(np.max(voxel_locs[:, 1])), int(np.min(voxel_locs[:, 1]))
        slice_z = int(punctum.z)

    except Exception:
        return {}, []

    x_size, y_size = 5, 5
    h, w = list(channel_map.values())[0].shape[1:]
    x1, x2 = max(0, x_min - x_size), min(w, x_max + x_size)
    y1, y2 = max(0, y_min - y_size), min(h, y_max + y_size)

    z_mask_map = {}
    detected_channels = []

    if excluded_channels is None:
        excluded_channels = set()

    try:
        for ch_num, ch_img in channel_map.items():
            if ch_num in excluded_channels:
                continue

            z_range = [z for z in range(slice_z - 1, slice_z + 2) if 0 <= z < ch_img.shape[0]]
            z_valid_masks = {}

            for z in z_range:
                try:
                    roi = ch_img[z, y1:y2, x1:x2]
                    if roi.size == 0 or np.max(roi) == 0:
                        continue

                    thresh = adaptive_thresholds.get(ch_num)
                    if thresh is None:
                        continue
                    roi_mask = (roi > thresh)

                    if ch_num == axon_dendrite_channel and axon_mask is not None:
                        if (0 <= z < axon_mask.shape[0] and
                            0 <= y1 < axon_mask.shape[1] and
                            0 <= x1 < axon_mask.shape[2] and
                            y2 <= axon_mask.shape[1] and
                            x2 <= axon_mask.shape[2]):
                            axon_roi = axon_mask[z, y1:y2, x1:x2]
                            roi_mask = roi_mask & (axon_roi > 0)

                    if np.any(roi_mask):
                        z_valid_masks[z] = (roi_mask, (y1, y2, x1, x2))

                except Exception:
                    continue

            if z_valid_masks:
                z_mask_map[ch_num] = z_valid_masks
                detected_channels.append(ch_num)

    except Exception:
        return {}, []

    return z_mask_map, detected_channels
