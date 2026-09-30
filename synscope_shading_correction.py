import os

from pathlib import PurePath

import cv2
import numpy as np

from zimg import *
from utils import img_util, shading_correction


def _tile_overlap_bounds(tile_a, tile_b):
    y0 = max(tile_a.start.y, tile_b.start.y)
    y1 = min(tile_a.end.y, tile_b.end.y)
    x0 = max(tile_a.start.x, tile_b.start.x)
    x1 = min(tile_a.end.x, tile_b.end.x)
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _estimate_blend_margins(tiles, tile_width, tile_height):
    margin_x = max(1, int(tile_width * 0.05))
    margin_y = max(1, int(tile_height * 0.05))
    for i, tile_a in enumerate(tiles):
        for tile_b in tiles[i + 1:]:
            bounds = _tile_overlap_bounds(tile_a, tile_b)
            if bounds is None:
                continue
            y0, y1, x0, x1 = bounds
            overlap_w = x1 - x0
            overlap_h = y1 - y0
            if overlap_w < tile_width and overlap_h >= tile_height * 0.5:
                margin_x = max(margin_x, overlap_w // 2)
            if overlap_h < tile_height and overlap_w >= tile_width * 0.5:
                margin_y = max(margin_y, overlap_h // 2)
    return margin_y, margin_x


def _tile_blend_weights(height, width, margin_y, margin_x):
    wy = np.ones(height, dtype=np.float64)
    wx = np.ones(width, dtype=np.float64)
    margin_y = min(margin_y, height // 2)
    margin_x = min(margin_x, width // 2)
    if margin_y > 0:
        ramp = np.linspace(0.0, 1.0, margin_y, endpoint=True)
        wy[:margin_y] = ramp
        wy[-margin_y:] = ramp[::-1]
    if margin_x > 0:
        ramp = np.linspace(0.0, 1.0, margin_x, endpoint=True)
        wx[:margin_x] = ramp
        wx[-margin_x:] = ramp[::-1]
    return wy[:, None] * wx[None, :]


def _refine_tile_gains_from_overlaps(tiles, corrected_tiles, init_gains, ds_ratio=1, n_iter=10):
    gains = np.array(init_gains, dtype=np.float64)
    for _ in range(n_iter):
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                bounds = _tile_overlap_bounds(tiles[i], tiles[j])
                if bounds is None:
                    continue
                y0, y1, x0, x1 = bounds
                roi_i = corrected_tiles[i][
                    (y0 - tiles[i].start.y) // ds_ratio:(y1 - tiles[i].start.y) // ds_ratio,
                    (x0 - tiles[i].start.x) // ds_ratio:(x1 - tiles[i].start.x) // ds_ratio,
                ]
                roi_j = corrected_tiles[j][
                    (y0 - tiles[j].start.y) // ds_ratio:(y1 - tiles[j].start.y) // ds_ratio,
                    (x0 - tiles[j].start.x) // ds_ratio:(x1 - tiles[j].start.x) // ds_ratio,
                ]
                if roi_i.size == 0 or roi_j.size == 0:
                    continue
                med_i = np.median(roi_i * gains[i])
                med_j = np.median(roi_j * gains[j])
                if med_i <= 1e-6 or med_j <= 1e-6:
                    continue
                ratio = np.sqrt(med_i / med_j)
                gains[i] /= ratio
                gains[j] *= ratio
    if gains[0] > 1e-6:
        gains /= gains[0]
    return gains.tolist()

def _estimate_tile_flatfield_and_gains(
    tile_images_ds,
    scene_tiles,
    tile_width,
    tile_height,
    working_size=512,
    gain_refine_iter=10,
):
    """Estimate shared tile flatfield and per-tile gains from downsampled tile images."""
    train_stack_ch = np.stack(tile_images_ds, axis=0).astype(np.float64)
    flatfield_ch, _ = shading_correction.BaSiC(
        train_stack_ch, estimate_darkfield=False, working_size=working_size,
    )
    flatfield_full = cv2.resize(
        flatfield_ch, dsize=(tile_width, tile_height), interpolation=cv2.INTER_CUBIC,
    )
    ff_floor = float(np.percentile(flatfield_full, 5))
    ff_floor_ds = float(np.percentile(flatfield_ch, 5))

    ntiles = len(tile_images_ds)
    gains = []
    corrected_ds_tiles = []
    for tile_img in tile_images_ds:
        ff_ds = cv2.resize(
            flatfield_ch,
            dsize=(tile_img.shape[1], tile_img.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        ff_ds = np.clip(ff_ds, ff_floor_ds, None)
        corrected_ds = tile_img.astype(np.float64) / ff_ds
        corrected_ds_tiles.append(corrected_ds)
        orig_ref = np.percentile(tile_img, 5)
        corr_ref = max(np.percentile(corrected_ds, 5), 1e-6)
        gains.append(float(orig_ref / corr_ref))

    ds_ratio = max(1, round(tile_width / corrected_ds_tiles[0].shape[1]))
    gains = _refine_tile_gains_from_overlaps(
        scene_tiles, corrected_ds_tiles, gains, ds_ratio=ds_ratio, n_iter=gain_refine_iter,
    )
    return (flatfield_full, ff_floor), gains


def shading_correction_convergence(
    img_file: str,
    result_folder: str = None,
    channels_to_correct: list = None,
    flatfield_mode: str = "per_z",
):
    """
    Apply shading correction to specified channels, leaving others unchanged.

    Args:
        img_file: Path to the CZI image file to process
        result_folder: Output folder for corrected images (default: same directory as img_file)
        channels_to_correct: List of channel indices to apply shading correction to (1-based indexing).
                            If None, all channels will be corrected (default behavior).
                            Example: [1, 2] to correct only channels 1 and 2.
        flatfield_mode: 'per_z' estimates flatfield and tile gains separately for each z-slice.
                        'shared' uses one flatfield/gain set pooled across all z-planes.
    """
    if flatfield_mode not in {"per_z", "shared"}:
        raise ValueError("flatfield_mode must be 'per_z' or 'shared'")
    if not os.path.exists(img_file):
        raise FileNotFoundError(f"Image file not found: {img_file}")

    if not result_folder:
        result_folder = os.path.dirname(img_file)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    filename = os.path.basename(img_file)
    img_info = ZImg.readImgInfos(img_file)
    print(img_info[0].depth, img_info[0].height, img_info[0].width)

    # Auto-detect dtype for this specific file
    sample_img = ZImg(img_file, scene=0, xRatio=4, yRatio=4)
    detected_dtype = sample_img.data[0].dtype

    if detected_dtype == np.uint8:
        input_dtype = 'uint8'
        max_pixel_value = 255
        final_dtype = np.uint8
    else:  # uint16 or other
        input_dtype = 'uint16'
        max_pixel_value = 65535
        final_dtype = np.uint16

    print(f"Auto-detected image dtype for {filename}: {input_dtype}")

    for scene_idx in range(len(img_info)):
        print(f'Running scene {scene_idx}')
        scene = int(scene_idx)
        blockList = ZImg.getInternalSubRegions(img_file)

        tile_width = blockList[scene][0].end.x - blockList[scene][0].start.x
        tile_height = blockList[scene][0].end.y - blockList[scene][0].start.y
        nchs = blockList[scene][0].end.c - blockList[scene][0].start.c
        ntiles = len(blockList[scene])

        flatfield = []
        tile_gains = [None] * img_info[0].numChannels
        stack_depth = img_info[scene].depth
        train_by_z = [
            [[None] * ntiles for _ in range(stack_depth)]
            for _ in range(img_info[0].numChannels)
        ]

        res_mask = np.zeros((img_info[scene].depth, img_info[scene].height, img_info[scene].width), dtype=np.uint8)

        for tile_idx, tile in enumerate(blockList[scene]):
            print(f'Running tile {tile_idx}')
            tile_img = ZImg(img_file, region=tile, scene=scene, xRatio=4, yRatio=4)
            img = tile_img.data[0].astype(input_dtype)

            img_chs = img.shape[0]
            img_depth = img.shape[1]

            for z_idx in range(img_depth):
                for ch in range(img_chs):
                    train_by_z[ch][z_idx][tile_idx] = img[ch, z_idx, :, :]

            res_mask[tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += 1

        res_mask[res_mask == 0] = 1
        scene_tiles = blockList[scene]
        blend_margin_y, blend_margin_x = _estimate_blend_margins(scene_tiles, tile_height, tile_width)
        print(f'Tile blend margins (y, x): {blend_margin_y}, {blend_margin_x}')
        tile_weights = _tile_blend_weights(tile_height, tile_width, blend_margin_y, blend_margin_x)
        if channels_to_correct is None:
            channels_to_correct_this_file = list(range(img_info[0].numChannels))
            channels_to_correct_display = list(range(1, img_info[0].numChannels + 1))
        else:
            # Validate channel numbers (1-based)
            max_channel = img_info[0].numChannels
            invalid_channels = [ch for ch in channels_to_correct if ch < 1 or ch > max_channel]
            if invalid_channels:
                raise ValueError(f"Invalid channel numbers: {invalid_channels}. Valid range is 1 to {max_channel}")
            # Convert to 0-based for internal processing
            channels_to_correct_this_file = [ch - 1 for ch in channels_to_correct]
            channels_to_correct_display = channels_to_correct

        print(f'Channels to correct (1-based): {channels_to_correct_display}')
        print(f'Flatfield mode: {flatfield_mode}')

        for ch in range(img_info[0].numChannels):
            if ch in channels_to_correct_this_file:
                if flatfield_mode == "per_z":
                    flatfield.append([])
                    tile_gains[ch] = []
                    for z_idx in range(img_depth):
                        print(f'Estimating shading for channel {ch + 1}, z {z_idx}')
                        ff, gains = _estimate_tile_flatfield_and_gains(
                            train_by_z[ch][z_idx],
                            scene_tiles,
                            tile_width,
                            tile_height,
                        )
                        flatfield[ch].append(ff)
                        tile_gains[ch].append(gains)
                        print(f'  z {z_idx} per-tile gains: {[round(g, 3) for g in gains]}')
                else:
                    pooled_tiles = [
                        train_by_z[ch][z_idx][tile_idx]
                        for z_idx in range(img_depth)
                        for tile_idx in range(ntiles)
                    ]
                    print(f'Estimating shared shading for channel {ch + 1} (all z pooled)')
                    train_stack_ch = np.stack(pooled_tiles, axis=0).astype(np.float64)
                    flatfield_ch, _ = shading_correction.BaSiC(
                        train_stack_ch, estimate_darkfield=False, working_size=512,
                    )
                    flatfield_full = cv2.resize(
                        flatfield_ch, dsize=(tile_width, tile_height), interpolation=cv2.INTER_CUBIC,
                    )
                    ff_floor = float(np.percentile(flatfield_full, 5))
                    ff_floor_ds = float(np.percentile(flatfield_ch, 5))
                    flatfield.append((flatfield_full, ff_floor))

                    gains = []
                    corrected_ds_tiles = []
                    for tile_idx in range(ntiles):
                        tile_gain_samples = []
                        tile_corrected_stack = []
                        for z_idx in range(img_depth):
                            tile_img = train_by_z[ch][z_idx][tile_idx]
                            ff_ds = cv2.resize(
                                flatfield_ch,
                                dsize=(tile_img.shape[1], tile_img.shape[0]),
                                interpolation=cv2.INTER_CUBIC,
                            )
                            ff_ds = np.clip(ff_ds, ff_floor_ds, None)
                            corrected_ds = tile_img.astype(np.float64) / ff_ds
                            tile_corrected_stack.append(corrected_ds)
                            orig_ref = np.percentile(tile_img, 5)
                            corr_ref = max(np.percentile(corrected_ds, 5), 1e-6)
                            tile_gain_samples.append(orig_ref / corr_ref)
                        corrected_ds_tiles.append(np.median(np.stack(tile_corrected_stack, axis=0), axis=0))
                        gains.append(float(np.median(tile_gain_samples)))
                    ds_ratio = max(1, round(tile_width / corrected_ds_tiles[0].shape[1]))
                    gains = _refine_tile_gains_from_overlaps(
                        scene_tiles, corrected_ds_tiles, gains, ds_ratio=ds_ratio,
                    )
                    tile_gains[ch] = gains
                    print(f'  per-tile gains for channel {ch + 1}: {[round(g, 3) for g in gains]}')
            else:
                flatfield.append(None)
                tile_gains[ch] = None

        whole_res_img = np.zeros((nchs, img_depth, img_info[scene].height, img_info[scene].width), dtype=np.float64)

        for z_idx in range(img_depth):
            print(f'Running {z_idx} slice')
            res_img = np.zeros((nchs, img_info[scene].height, img_info[scene].width), dtype=np.float64)

            for ch in range(nchs):
                print(f'Running channel {ch + 1} (1-based)')
                weight_sum = np.zeros((img_info[scene].height, img_info[scene].width), dtype=np.float64)
                for tile_idx, tile in enumerate(scene_tiles):
                    print(f'Running tile {tile_idx}')
                    tile_img = ZImg(img_file, region=tile, scene=scene)
                    img = tile_img.data[0].astype(input_dtype)
                    img_ch = img[ch, z_idx, :, :].astype(np.float64)

                    if ch in channels_to_correct_this_file:
                        if flatfield_mode == "per_z":
                            flatfield_ch, ff_floor = flatfield[ch][z_idx]
                            tile_gain = tile_gains[ch][z_idx][tile_idx]
                        else:
                            flatfield_ch, ff_floor = flatfield[ch]
                            tile_gain = tile_gains[ch][tile_idx]
                        was_saturated = img_ch == max_pixel_value
                        safe_flatfield = np.clip(flatfield_ch, ff_floor, None)
                        corrected_block = (img_ch / safe_flatfield) * tile_gain
                        corrected_block[was_saturated] = max_pixel_value
                    else:
                        corrected_block = img_ch

                    y0, y1 = tile.start.y, tile.end.y
                    x0, x1 = tile.start.x, tile.end.x
                    res_img[ch, y0:y1, x0:x1] += corrected_block * tile_weights
                    weight_sum[y0:y1, x0:x1] += tile_weights

                res_img[ch] /= np.maximum(weight_sum, 1e-6)

            res_img = np.clip(res_img, 0, max_pixel_value).astype(final_dtype)
            whole_res_img[:, z_idx, :, :] = res_img

        whole_res_img = whole_res_img.astype(final_dtype)
        output_path = os.path.join(result_folder, f'{PurePath(filename).stem}_shading_corrected.tiff')
        print(f'Saving {filename} shading correction to {output_path}')
        img_util.write_img(
            filename=output_path,
            img_data=whole_res_img,
            voxel_size_unit=img_info[scene].voxelSizeUnit,
            voxel_size_x=img_info[scene].voxelSizeX,
            voxel_size_y=img_info[scene].voxelSizeY,
            voxel_size_z=img_info[scene].voxelSizeZ,
        )

if __name__ == "__main__":

    img_file = "path/to/sample.czi"
    shading_correction_convergence(img_file, channels_to_correct=None)
