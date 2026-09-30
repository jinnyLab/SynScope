#!/usr/bin/env python3
"""
Core functions for mGRASP puncta classification.

This module contains essential functions for data loading, thresholding, and axon/dendrite separation
used by the ML-based classification pipeline.
"""

import os

from pathlib import Path,PurePath

from zimg import *

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import pandas as pd
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


def apply_channel_threshold_multipliers(
    adaptive_thresholds: Dict[int, float],
    multipliers: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """Scale per-channel adaptive thresholds by optional multipliers."""
    if not multipliers:
        return adaptive_thresholds
    return {
        ch: float(thr) * float(multipliers.get(ch, 1.0))
        for ch, thr in adaptive_thresholds.items()
    }


def channel_intensity_threshold_dataframe(
    base_thresholds: Dict[int, float],
    final_thresholds: Dict[int, float],
    multipliers: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Build a per-channel report of intensity detection thresholds."""
    rows = []
    for ch in sorted(final_thresholds):
        multiplier = float(multipliers.get(ch, 1.0)) if multipliers else 1.0
        rows.append(
            {
                "channel": ch,
                "base_threshold": float(base_thresholds.get(ch, final_thresholds[ch])),
                "multiplier": multiplier,
                "final_threshold": float(final_thresholds[ch]),
            }
        )
    return pd.DataFrame(rows)


def print_channel_intensity_thresholds(
    base_thresholds: Dict[int, float],
    final_thresholds: Dict[int, float],
    multipliers: Optional[Dict[int, float]] = None,
) -> None:
    """Print per-channel intensity thresholds used for signal detection."""
    print("  Channel intensity thresholds:")
    for ch in sorted(final_thresholds):
        multiplier = float(multipliers.get(ch, 1.0)) if multipliers else 1.0
        base = float(base_thresholds.get(ch, final_thresholds[ch]))
        final = float(final_thresholds[ch])
        print(
            f"    Channel {ch}: final={final:.4f} "
            f"(base={base:.4f}, multiplier={multiplier:.3f})"
        )


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

_X_COLUMN_ALIASES = frozenset({"x", "punctum x", "punctumx", "coord x"})
_Y_COLUMN_ALIASES = frozenset({"y", "punctum y", "punctumy", "coord y"})
_Z_COLUMN_ALIASES = frozenset({"z", "punctum z", "punctumz", "coord z"})


@dataclass
class CsvPunctum:
    """Minimal punctum object constructed from x, y, z coordinates."""
    x: float
    y: float
    z: float
    voxelLocations: np.ndarray

    @classmethod
    def from_coordinates(
        cls,
        x: float,
        y: float,
        z: float,
        voxel_patch_size: int = 5,
    ) -> "CsvPunctum":
        xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
        half_low = voxel_patch_size // 2
        offsets = range(-half_low, voxel_patch_size - half_low)
        voxels = [
            [xi + dx, yi + dy, zi]
            for dx in offsets
            for dy in offsets
        ]
        return cls(
            x=float(x),
            y=float(y),
            z=float(z),
            voxelLocations=np.array(voxels, dtype=np.float64),
        )

    def to_zpunctum(self):
        """Convert to a zimg ZPunctum for .nimp export."""
        punctum = ZPunctum()
        voxels = np.asarray(self.voxelLocations, dtype=np.int32)
        punctum.voxelLocations = voxels
        punctum.voxelIntensities = np.ones(len(voxels), dtype=np.float32)
        punctum.updateFromVoxelsList()
        return punctum


def _normalize_column_name(column: str) -> str:
    return column.strip().lower().replace("_", " ")


def _resolve_xyz_columns(columns) -> Tuple[str, str, str]:
    x_col = y_col = z_col = None
    for col in columns:
        normalized = _normalize_column_name(col)
        if normalized in _X_COLUMN_ALIASES:
            x_col = col
        elif normalized in _Y_COLUMN_ALIASES:
            y_col = col
        elif normalized in _Z_COLUMN_ALIASES:
            z_col = col
    if x_col is None or y_col is None or z_col is None:
        found = ", ".join(str(c) for c in columns)
        raise ValueError(
            "CSV must contain x, y, and z coordinate columns. "
            "Accepted names include: x/y/z or punctum x/punctum y/punctum z. "
            f"Found columns: {found}"
        )
    return x_col, y_col, z_col


def _load_coordinate_csv(csv_path: str) -> pd.DataFrame:
    """Load a coordinate CSV, skipping a leading title row when present."""
    df = pd.read_csv(csv_path)
    if not df.empty:
        try:
            _resolve_xyz_columns(df.columns)
            return df
        except ValueError:
            pass

    for skiprows in (1, 2):
        df_candidate = pd.read_csv(csv_path, skiprows=skiprows)
        if df_candidate.empty:
            continue
        try:
            _resolve_xyz_columns(df_candidate.columns)
            print(f"[INFO] Skipped {skiprows} leading row(s) in CSV header: {csv_path}")
            return df_candidate
        except ValueError:
            continue

    found = ", ".join(str(c) for c in df.columns)
    raise ValueError(
        "CSV must contain x, y, and z coordinate columns. "
        "Accepted names include: x/y/z or punctum x/punctum y/punctum z. "
        f"Found columns: {found}"
    )


def resolve_voxel_patch_size(
    voxel_patch_size: Optional[int] = None,
    voxel_patch_radius: Optional[int] = None,
) -> int:
    """
    Resolve square voxel patch side length from size or radius.

    Examples: size 5 or radius 2 -> 5x5 patch; size 10 -> 10x10 patch.
    """
    if voxel_patch_size is not None and voxel_patch_radius is not None:
        raise ValueError("Specify only one of voxel_patch_size or voxel_patch_radius.")

    if voxel_patch_size is not None:
        if voxel_patch_size < 1:
            raise ValueError("voxel_patch_size must be >= 1.")
        return voxel_patch_size

    if voxel_patch_radius is not None:
        if voxel_patch_radius < 0:
            raise ValueError("voxel_patch_radius must be >= 0.")
        return 2 * voxel_patch_radius + 1

    return 5


def read_puncta_coordinates_csv(
    csv_path: str,
    voxel_patch_size: int = 5,
) -> List[CsvPunctum]:
    """Load puncta coordinates from a CSV file."""
    df = _load_coordinate_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    x_col, y_col, z_col = _resolve_xyz_columns(df.columns)
    print(f"[INFO] Using CSV columns: x={x_col}, y={y_col}, z={z_col}")
    punctum_list: List[CsvPunctum] = []

    for row in df[[x_col, y_col, z_col]].itertuples(index=False, name=None):
        x, y, z = row
        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            continue
        punctum_list.append(
            CsvPunctum.from_coordinates(float(x), float(y), float(z), voxel_patch_size=voxel_patch_size)
        )

    if not punctum_list:
        raise ValueError(f"No valid x, y, z coordinates found in CSV: {csv_path}")

    print(f"[INFO] Loaded {len(punctum_list)} puncta from CSV: {csv_path}")
    print(f"[INFO] Voxel patch size: {voxel_patch_size}x{voxel_patch_size}")
    return punctum_list


def _load_image_channels(
    img_folder: str,
    img_name: str,
    mgrasp_channel: Optional[int] = None,
    axon_dendrite_channel: Optional[int] = None,
    use_axon_dendrite: bool = True,
) -> Tuple[str, Dict[int, np.ndarray], Optional[np.ndarray], Dict[int, np.ndarray]]:
    """Load multi-channel image data and return analysis channel maps."""
    img_path = os.path.join(img_folder, img_name)
    img_infos = ZImg.readImgInfos(img_path)
    num_image_planes = img_infos[0].numChannels

    if mgrasp_channel is not None:
        if mgrasp_channel < 1 or mgrasp_channel > num_image_planes:
            raise ValueError(f"mGRASP channel {mgrasp_channel} is out of range. Image has {num_image_planes} channels.")

    if axon_dendrite_channel is not None:
        if axon_dendrite_channel < 1 or axon_dendrite_channel > num_image_planes:
            raise ValueError(
                f"Axon/dendrite channel {axon_dendrite_channel} is out of range. "
                f"Image has {num_image_planes} channels."
            )
        if mgrasp_channel is not None and axon_dendrite_channel == mgrasp_channel:
            raise ValueError(
                f"Axon/dendrite channel cannot be the same as mGRASP channel ({mgrasp_channel})."
            )

    imgObj = ZImg(img_path, scene=0, xRatio=1, yRatio=1)
    img = imgObj.data[0]
    if img.max() > 255:
        img = np.asarray((img / img.max()) * 255.0, dtype=np.float32)

    # Copy channel arrays so data remain valid after imgObj is released.
    all_channels = {
        i + 1: np.array(img[i], copy=True)
        for i in range(num_image_planes)
    }
    if mgrasp_channel is None:
        channel_map = dict(all_channels)
    else:
        channel_map = {ch: img_data for ch, img_data in all_channels.items() if ch != mgrasp_channel}

    axon_dendrite_image = None
    if axon_dendrite_channel is not None:
        axon_dendrite_image = all_channels[axon_dendrite_channel]

    if not use_axon_dendrite and axon_dendrite_channel is not None:
        channel_map.pop(axon_dendrite_channel, None)

    if not channel_map:
        raise ValueError("No channels available for analysis.")

    return img_name, channel_map, axon_dendrite_image, all_channels


def _sample_roi_intensities(
    punctum_list: List,
    channel_map: Dict[int, np.ndarray],
) -> Dict[int, List[float]]:
    """Sample ROI intensities around puncta for adaptive thresholding."""
    roi_intensity_samples = {ch: [] for ch in channel_map}
    x_size, y_size = 5, 5

    if len(punctum_list) == 0:
        return roi_intensity_samples

    for punctum in tqdm(punctum_list, desc="Sampling intensities from puncta"):
        vl = getattr(punctum, 'voxelLocations', None)
        if vl is None or getattr(vl, 'size', 0) == 0 or vl.shape[1] < 2:
            continue
        x_max, x_min = int(np.max(vl[:, 0])), int(np.min(vl[:, 0]))
        y_max, y_min = int(np.max(vl[:, 1])), int(np.min(vl[:, 1]))
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
                # Use mean intensity per punctum to avoid unsafe uint8->float64
                # casts on large zimg-backed array views.
                roi_intensity_samples[ch_num].append(float(np.mean(roi, dtype=np.float32)))

    return roi_intensity_samples


def load_data_from_csv(
    img_folder: str,
    img_name: str,
    csv_path: str,
    axon_dendrite_channel: Optional[int] = None,
    use_axon_dendrite: bool = True,
    voxel_patch_size: Optional[int] = None,
    voxel_patch_radius: Optional[int] = None,
    mgrasp_channel: Optional[int] = None,
):
    """Load image data and puncta coordinates from a CSV file.

    Args:
        img_folder: Path to folder containing the image file
        img_name: Name of the image file
        csv_path: Path to CSV with x, y, z coordinate columns
        axon_dendrite_channel: Channel to use for axon/dendrite morphology analysis (optional)
        use_axon_dendrite: If True, axon_dendrite_channel is included in analysis; if False, excluded
        voxel_patch_size: Side length of square voxel patch (odd integer, e.g. 5 for 5x5)
        voxel_patch_radius: Patch radius in pixels (patch size = 2 * radius + 1)
        mgrasp_channel: Optional mGRASP channel to exclude. If None, all channels are used.

    Returns:
        Tuple of (img_name, channel_map, axon_dendrite_image, punctum_list, roi_intensity_samples)
    """
    patch_size = resolve_voxel_patch_size(
        voxel_patch_size=voxel_patch_size,
        voxel_patch_radius=voxel_patch_radius,
    )
    img_name, channel_map, axon_dendrite_image, _ = _load_image_channels(
        img_folder,
        img_name,
        mgrasp_channel=mgrasp_channel,
        axon_dendrite_channel=axon_dendrite_channel,
        use_axon_dendrite=use_axon_dendrite,
    )
    punctum_list = read_puncta_coordinates_csv(csv_path, voxel_patch_size=patch_size)
    roi_intensity_samples = _sample_roi_intensities(punctum_list, channel_map)
    return img_name, channel_map, axon_dendrite_image, punctum_list, roi_intensity_samples


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
    img_name, channel_map, axon_dendrite_image, _ = _load_image_channels(
        img_folder,
        img_name,
        mgrasp_channel=mgrasp_channel,
        axon_dendrite_channel=axon_dendrite_channel,
        use_axon_dendrite=use_axon_dendrite,
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

    roi_intensity_samples = _sample_roi_intensities(punctum_list, channel_map)
    return img_name, channel_map, axon_dendrite_image, punctum_list, roi_intensity_samples


# =========================
# Utility Functions
# =========================

def normalize_channel_combination(channels: Union[List[int], Tuple[int, ...]]) -> str:
    """Convert detected channel numbers to a canonical prediction label (e.g. [3, 1, 2] -> '1_2_3')."""
    if not channels:
        return "low_confidence"
    unique_channels = sorted({int(ch) for ch in channels})
    return "_".join(str(ch) for ch in unique_channels)


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
