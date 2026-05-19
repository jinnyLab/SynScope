#!/usr/bin/env python3
import os
import sys

from pathlib import Path,PurePath
sys.path.append('../')

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
