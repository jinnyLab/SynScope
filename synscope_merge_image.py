#!/usr/bin/env python3
import os

from pathlib import Path,PurePath

import numpy as np
import tifffile

from zimg import *
from skimage.transform import resize
from skimage.util import img_as_uint
from utils import img_util

def merge_channel(image_folder: str):
    (_, _, file_list) = next(os.walk(image_folder))
    img_list = [fn for fn in file_list if '.tiff' in fn]
    img_list.sort()

    print(img_list)

    base_name = img_list[0].rsplit('_', 1)[0] + '_merged.tiff'
    combined_filename = os.path.join(image_folder, base_name)

    imgMerge = ZImgMerge()
    imgSubBlocks = []

    for idx, fn in enumerate(img_list):
        imgSubBlocks.append(ZImgTileSubBlock(ZImgSource(os.path.join(image_folder, fn))))
        imgMerge.addImg(imgSubBlocks[-1], (0, 0, 0, idx, 0), PurePath(fn).name)

    imgMerge.resolveLocations()
    imgMerge.save(combined_filename)
    print(f'image {combined_filename} done')

if __name__ == '__main__':
    image_folder = "path/to/images"
    merge_channel(image_folder=image_folder)