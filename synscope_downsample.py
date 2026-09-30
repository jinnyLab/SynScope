#!/usr/bin/env python3
import os

from pathlib import Path,PurePath

import numpy as np
import tifffile

from zimg import *
from skimage.transform import resize
from skimage.util import img_as_uint
from utils import img_util


def downsample_zimg(image_folder: str, filename: str, ratio_x: int, ratio_y: int) -> np.ndarray:
    imgObj = ZImg(os.path.join(image_folder, filename), scene=0, xRatio=ratio_x, yRatio=ratio_y)
    img_data = imgObj.data[0]
    img_dtype = img_data.dtype
    img_data = img_data.astype(img_dtype)

    base_name = filename.rsplit('_', 1)[0] + '_downsampled.tiff'
    output_name = os.path.join(image_folder,base_name)
    img_util.write_img(filename=output_name, img_data=img_data)


if __name__ == '__main__':
    image_folder = "path/to/images"
    filename = "sample.tiff"
    ratio_x = 2
    ratio_y = 2
    downsample_zimg(image_folder=image_folder, filename=filename, ratio_x=ratio_x, ratio_y=ratio_y)