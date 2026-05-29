import os
import sys

from pathlib import Path, PurePath
sys.path.append('../')

import ants
import numpy as np

from zimg import ZImg
from utils import img_util

TX_LIST_LSM780 = [
    './model/_chromatic_shift_parameters/lsm780/lsm780_chromatic_shift.mat'
]

TX_LIST_LSM980 = [
    './model/_chromatic_shift_parameters/lsm980/lsm980_chromatic_shift.nii.gz',
    './model/_chromatic_shift_parameters/lsm980/lsm980_chromatic_shift.mat'
]


def _infer_dtype(img_volume: np.ndarray) -> str:
    if img_volume.dtype == np.uint8:
        return 'uint8'
    if img_volume.dtype == np.uint16:
        return 'uint16'
    if np.issubdtype(img_volume.dtype, np.integer):
        return 'uint8' if img_volume.max() <= np.iinfo(np.uint8).max else 'uint16'
    if np.issubdtype(img_volume.dtype, np.floating):
        return 'uint8' if img_volume.max() <= np.iinfo(np.uint8).max else 'uint16'
    raise ValueError(f"Unsupported image dtype for inference: {img_volume.dtype}")


def apply_chromatic_correction(
    img_path: str,
    scope: str = 'lsm980',
    output_suffix: str = '_chromatic_corrected',
    dtype: str = 'auto',
    moving_channel: int = 4
):

    if scope not in ['lsm980', 'lsm780', 'calculate']:
        raise ValueError("scope must be 'lsm780', 'lsm980', or 'calculate'")

    img_volume_ZImg = ZImg(img_path)
    img_volume = img_volume_ZImg.data[0]
    if dtype is None:
        dtype = 'auto'
    dtype = dtype.lower()
    detected_dtype = _infer_dtype(img_volume) if dtype == 'auto' else dtype

    if detected_dtype not in ['uint8', 'uint16']:
        raise ValueError("dtype must be 'auto', 'uint8', or 'uint16'")

    final_dtype = np.uint8 if detected_dtype == 'uint8' else np.uint16

    fixed_ch = 0

    if moving_channel not in [1, 2, 3, 4, 5]:
        raise ValueError("moving_channel must be one of 1, 2, 3, 4, or 5")

    # channel indices are 1-based; convert to 0-based index.
    moving_ch = moving_channel - 1

    if moving_ch >= img_volume.shape[0]:
        raise ValueError(
            f"moving_channel={moving_channel} is out of range for input with {img_volume.shape[0]} channels"
        )

    fixed = ants.from_numpy(img_volume[fixed_ch].astype('float32' if detected_dtype == 'uint8' else 'float64'))
    moving = ants.from_numpy(img_volume[moving_ch].astype('float32' if detected_dtype == 'uint8' else 'float64'))

    if scope == 'calculate':
        # Calculate transformation using registration
        mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine', restrict_deformation='1x0x0x0x1x0x0x0x1x1x1x1')
        moved = mytx['warpedmovout']
    else:
        # Use predefined transformation parameters
        if scope == 'lsm980':
            tx_list = TX_LIST_LSM980
        elif scope == 'lsm780':
            tx_list = TX_LIST_LSM780

        moved = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx_list)

    img_volume[moving_ch] = moved.numpy().astype(final_dtype)

    output_name = PurePath(img_path).name.replace('_shading_corrected', output_suffix)
    output_path = os.path.join(PurePath(img_path).parent, output_name)

    img_util.write_img(filename=output_path, img_data=img_volume)
    print(f"Saved corrected image to: {output_path}")

if __name__ == "__main__":

    img_folder = 'path/to/image/folder'
    filename = 'image_name.tiff'
    img_path = os.path.join(img_folder, filename)

    moving_channel = 4  # Choose from 1, 2, 3, 4, or 5

    apply_chromatic_correction(img_path, scope='calculate', moving_channel=moving_channel)
