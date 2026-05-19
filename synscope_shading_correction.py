import os
import sys

from pathlib import Path, PurePath
sys.path.append('../')

import cv2
import numpy as np

from zimg import *
from utils import shading_correction

def shading_correction_convergence(img_file: str, result_folder: str = None, channels_to_correct: list = None):
    """
    Apply shading correction to specified channels, leaving others unchanged.

    Args:
        img_file: Path to the CZI image file to process
        result_folder: Output folder for corrected images (default: same directory as img_file)
        channels_to_correct: List of channel indices to apply shading correction to (1-based indexing).
                            If None, all channels will be corrected (default behavior).
                            Example: [1, 2] to correct only channels 1 and 2.
    """
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
        train_stack = [[] for _ in range(img_info[0].numChannels)]

        res_mask = np.zeros((img_info[scene].depth, img_info[scene].height, img_info[scene].width), dtype=np.uint8)

        for tile_idx, tile in enumerate(blockList[scene]):
            print(f'Running tile {tile_idx}')
            tile_img = ZImg(img_file, region=tile, scene=scene, xRatio=4, yRatio=4)
            img = tile_img.data[0].astype(input_dtype)

            img_chs = img.shape[0]
            img_depth = img.shape[1]
            img_height = img.shape[2]

            for z_idx in range(img_depth):
                for ch in range(img_chs):
                    train_stack[ch].append(img[ch, z_idx, :, :])

            res_mask[tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += 1

        res_mask[res_mask == 0] = 1


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

        # Only estimate flatfield for channels that need correction
        for ch in range(img_info[0].numChannels):
            if ch in channels_to_correct_this_file:
                train_stack_ch = np.dstack(train_stack[ch])
                train_stack_ch = np.moveaxis(train_stack_ch, -1, 0).copy(order='C')

                print(f'Estimating shading parameter for channel {ch + 1} (1-based)')
                flatfield_ch, _ = shading_correction.BaSiC(train_stack_ch, estimate_darkfield=False, working_size=img_height)
                flatfield_ch = cv2.resize(flatfield_ch, dsize=(tile_width, tile_height), interpolation=cv2.INTER_CUBIC)
                flatfield.append(flatfield_ch)
            else:
                flatfield.append(None)  # No flatfield needed for uncorrected channels

        whole_res_img = np.zeros((nchs, img_depth, img_info[scene].height, img_info[scene].width), dtype=np.float64)

        for z_idx in range(img_depth):
            print(f'Running {z_idx} slice')
            res_img = np.zeros((nchs, img_info[scene].height, img_info[scene].width), dtype=np.float64)

            for ch in range(nchs):
                print(f'Running channel {ch + 1} (1-based)')
                for tile_idx, tile in enumerate(blockList[scene]):
                    print(f'Running tile {tile_idx}')
                    tile_img = ZImg(img_file, region=tile, scene=scene)
                    img = tile_img.data[0].astype(input_dtype)
                    img_ch = img[ch, z_idx, :, :].astype(np.float64)

                    if ch in channels_to_correct_this_file:
                        # Apply shading correction

                        corrected_block = img_ch / flatfield[ch]

                        saturated_mask = img_ch == max_pixel_value
                        corrected_block[saturated_mask] = max_pixel_value
                        res_img[ch, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += corrected_block
                    else:
                        # Keep original image without correction
                        res_img[ch, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += img_ch

                res_img[ch] /= res_mask[0]

            res_img = np.clip(res_img, 0, max_pixel_value).astype(final_dtype)
            whole_res_img[:, z_idx, :, :] = res_img

        whole_res_img = whole_res_img.astype(final_dtype)
        print(f'Saving {filename} shading correction')
        img = ZImg(whole_res_img, img_info[scene])
        img.save(os.path.join(result_folder, f'{PurePath(filename).stem}_shading_corrected.tiff'))

if __name__ == "__main__":

    img_file = 'path/to/image'
    shading_correction_convergence(img_file, channels_to_correct=None)
