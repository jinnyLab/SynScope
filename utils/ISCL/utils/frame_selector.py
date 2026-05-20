#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile


def load_3d_image(image_path, channel=None, channel_axis=1):
    """
    Load a 3D image (Z, Y, X) or a single channel from a 4D stack using tifffile.
    """
    arr = tifffile.imread(image_path)

    if arr.ndim == 3:
        img = arr
    elif arr.ndim == 4:
        # Normalize channel_axis and move channels to axis 0 → (C, ...).
        channel_axis = channel_axis % arr.ndim
        c_first = np.moveaxis(arr, channel_axis, 0)
        num_channels = c_first.shape[0]

        # Use 1-based channel indexing externally (1..C), convert to 0-based internally.
        if channel is None:
            channel = 1
        if not (1 <= channel <= num_channels):
            raise ValueError(f"Requested channel {channel} out of valid range [1, {num_channels}] for shape {arr.shape}")
        ch_idx = channel - 1

        img = c_first[ch_idx]
    else:
        raise ValueError(f"Unexpected image shape from {image_path}: {arr.shape}")

    # Normalize to uint8-like range if dynamic range exceeds 255
    img = img.astype(np.float32)
    vmax = img.max()
    if vmax > 255:
        img = img / vmax * 255.0
    img = img.astype(np.uint8)

    if img.ndim != 3:
        raise ValueError(f"Expected 3D image (Z,Y,X) after loading; got shape {img.shape}")

    return img


def percentile_normalize(values, p_low=2, p_high=98):
    """Percentile normalization to [0, 1]."""
    lo, hi = np.percentile(values, (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(values)
    norm = (values - lo) / (hi - lo)
    return np.clip(norm, 0, 1)


def mean_intensity_per_slice(image_3d, normalize=True):
    """Compute mean intensity for each z-slice; optional percentile normalization."""
    if image_3d.ndim != 3:
        raise ValueError("Expected 3D image (Z, Y, X)")
    means = image_3d.mean(axis=(1, 2), dtype=np.float64)
    return percentile_normalize(means) if normalize else means


def top_two_peak_centers(intensities):
    """Return indices of the two highest peaks (slice indices of top two maxima)."""
    if intensities.size == 0:
        return []
    return np.argsort(intensities)[::-1][:2].tolist()


def slices_around_center(center, total, count=5):
    """Return up to `count` consecutive indices centered at `center` (best-effort at edges)."""
    half = count // 2
    start = max(0, center - half)
    end = min(total, start + count)
    start = max(0, end - count)
    return list(range(start, end))


def valid_z_range(intensities, threshold=0.05):
    """
    Suggest a contiguous Z-range [z_start, z_end) by trimming low-intensity
    slices at the beginning and end of the stack.
    """
    if intensities.size == 0:
        return 0, 0
    # intensities are percentile-normalized to [0,1]; trim where they stay near 0
    above = np.where(intensities > threshold)[0]
    if above.size == 0:
        return 0, len(intensities)
    z_start = int(above[0])
    z_end = int(above[-1]) + 1  # end is exclusive
    return z_start, z_end


def find_clean_noisy_reference(image_path, channel=None):
    """
    Compute selections and return:
      - clean: list of 10 slice indices
      - noisy: list of 10 slice indices
      - reference: list with 1 index
      - z_range: (z_start, z_end) suggested usable Z-range based on intensity histogram
    """
    img = load_3d_image(image_path, channel=channel)
    per_slice_mean = mean_intensity_per_slice(img, normalize=True)
    z_start, z_end = valid_z_range(per_slice_mean)

    # Two highest peak slices (centers)
    peak_centers = top_two_peak_centers(per_slice_mean)
    if len(peak_centers) == 0:
        order = np.argsort(per_slice_mean)[::-1]
        clean = sorted(order[:10].tolist())
        mid_center = len(per_slice_mean) // 2
        noisy = slices_around_center(mid_center, len(per_slice_mean), count=10)
        reference = [int(order[0])]
        return clean, noisy, reference, (z_start, z_end)

    if len(peak_centers) == 1:
        first = peak_centers[0]
        others = [i for i in np.argsort(per_slice_mean)[::-1] if i != first]
        second = others[0] if others else first
    else:
        first, second = peak_centers[0], peak_centers[1]

    # 5 around each of the two peaks => aim for exactly 10 "clean" slices
    clean_first = slices_around_center(first, len(per_slice_mean), count=5)
    clean_second = slices_around_center(second, len(per_slice_mean), count=5)

    # Start with unique indices from the two 5-slice windows
    clean = []
    for idx in clean_first + clean_second:
        if idx not in clean:
            clean.append(idx)

    # If overlap reduced the count below 10, fill from global intensity ranking (excluding already chosen)
    if len(clean) < 10:
        ranked = np.argsort(per_slice_mean)[::-1]
        for idx in ranked:
            if idx not in clean:
                clean.append(int(idx))
            if len(clean) >= 10:
                break

    # If we somehow exceeded 10 due to edge conditions, truncate while keeping sorted order
    clean = sorted(clean[:10])

    # 10 around middle => "noisy"
    mid_center = len(per_slice_mean) // 2
    noisy = slices_around_center(mid_center, len(per_slice_mean), count=10)

    # reference: highest intensity within the second peak neighborhood (best slice at/near second)
    ref_window = slices_around_center(second, len(per_slice_mean), count=5)
    ref_local_idx = int(np.argmax(per_slice_mean[ref_window]))
    reference = [ref_window[ref_local_idx]]

    return clean, noisy, reference, (z_start, z_end)


def export_csv(image_path, output_csv, clean_slices, noisy_slices, reference_slice, z_range=None):
    rows = []
    # Store slice indices as 1-based, comma-separated strings for readability, e.g. "1,2,3,4".
    clean_str = ",".join(str(int(s) + 1) for s in clean_slices)
    noisy_str = ",".join(str(int(s) + 1) for s in noisy_slices)
    ref_idx_1 = int(reference_slice[0]) + 1

    rows.append({"frame_type": "clean frame", "slice_index": clean_str})
    rows.append({"frame_type": "noisy frame", "slice_index": noisy_str})
    rows.append({"frame_type": "reference frame", "slice_index": ref_idx_1})

    # Optionally include suggested Z-range as "Target range"
    if z_range is not None:
        z_start, z_end = z_range  # internal indices: start inclusive, end exclusive (0-based)
        # Convert to 1-based, inclusive range for readability, e.g. "1,20"
        start_1 = int(z_start) + 1
        end_1 = int(z_end)
        rows.append({"frame_type": "Target range", "slice_index": f"{start_1},{end_1}"})
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    test_image_path = \
        "/Volumes/shared/Personal/Yoonkyoung/2A/denoising_test/0_merge_split_test/JK1205_2_8_MTS1_Airyscan_Processing_5_downsampled.tiff"

    clean, noisy, reference, z_range = find_clean_noisy_reference(test_image_path, channel=4)

    out_csv = f"{Path(test_image_path).stem}_frame_selection.csv"
    export_csv(test_image_path, out_csv, clean, noisy, reference, z_range=z_range)
    print('done')
