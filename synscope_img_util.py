#!/usr/bin/env python3
import os

from pathlib import Path,PurePath

import numpy as np

from zimg import *
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

    channels = []
    voxel_kwargs = {}
    for fn in img_list:
        img = ZImg(os.path.join(image_folder, fn))
        data = img.data[0]
        if data.ndim == 4:
            if data.shape[0] != 1:
                raise ValueError(
                    f'{fn} has {data.shape[0]} channels; merge expects one channel per file'
                )
            data = data[0]
        channels.append(data)
        if not voxel_kwargs:
            info = img.info
            voxel_kwargs = dict(
                voxel_size_unit=info.voxelSizeUnit,
                voxel_size_x=info.voxelSizeX,
                voxel_size_y=info.voxelSizeY,
                voxel_size_z=info.voxelSizeZ,
            )

    merged = np.stack(channels, axis=0)
    img_util.write_img(filename=combined_filename, img_data=merged, **voxel_kwargs)
    print(f'image {combined_filename} done')
