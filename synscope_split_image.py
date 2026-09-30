#!/usr/bin/env python3
import os

from pathlib import Path,PurePath

import numpy as np
import tifffile

from zimg import *
from skimage.transform import resize
from skimage.util import img_as_uint
from utils import img_util


def channel_split(image_folder: str, filename: str, result_folder: str = None):
    input_path = os.path.join(image_folder, filename)

    if not result_folder:
        result_folder = os.path.join(image_folder, 'channel_split')
    os.makedirs(result_folder, exist_ok=True)

    img_obj = ZImg(input_path, scene=0, xRatio=1, yRatio=1)
    img_infos = ZImg.readImgInfos(input_path)
    img = img_obj.data[0]

    inferred_dtype = img.dtype

    for ch in range(img_infos[0].numChannels):
        ch_img = img[ch, :, :, :].astype(inferred_dtype)
        output_name = os.path.join(result_folder, f'{PurePath(filename).stem}_ch{ch+1}.tiff')
        img_util.write_img(filename=output_name, img_data=ch_img)
        print(ch_img.shape)
    print('Channel split completed.')

if __name__ == '__main__':

    image_folder = "path/to/images"
    filename = "sample.tiff"
    
    channel_split(image_folder=image_folder, filename=filename)