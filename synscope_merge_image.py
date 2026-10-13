#!/usr/bin/env python3
import os

import numpy as np

from zimg import *
from utils import img_util

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

if __name__ == '__main__':
    image_folder = "path/to/images"
    merge_channel(image_folder=image_folder)