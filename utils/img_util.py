from zimg import *


def write_img(filename: str, img_data: np.ndarray, *,
              des_channel: int = 0, des_depth: int = 0, des_height: int = 0, des_width: int = 0,
              pad_before_ratio: float = 0.5,
              voxel_size_unit: VoxelSizeUnit = VoxelSizeUnit.none,
              voxel_size_x: float = 1., voxel_size_y: float = 1.,voxel_size_z: float = 1.):
    """
    img_data: 1-4 dimensions ndarray (C x D x H) x W
    pad or remove before and after to match des dimension if des_* > 0
    """
    assert img_data.ndim > 0 and img_data.size > 0, img_data.shape
    if img_data.ndim == 1:
        img_data = img_data[np.newaxis, np.newaxis, np.newaxis, :]
    elif img_data.ndim == 2:
        img_data = img_data[np.newaxis, np.newaxis, :, :]
    elif img_data.ndim == 3:
        img_data = img_data[np.newaxis, :, :, :]
    else:
        assert img_data.ndim == 4, img_data.shape

    pad_data = np.ascontiguousarray(
        pad_img(img_data, des_channel=des_channel, des_depth=des_depth, des_height=des_height,
                des_width=des_width, pad_before_ratio=pad_before_ratio))

    info = ZImgInfo()
    info.voxelSizeUnit = voxel_size_unit
    info.voxelSizeX = voxel_size_x
    info.voxelSizeY = voxel_size_y
    info.voxelSizeZ = voxel_size_z
    res_img = ZImg(pad_data, info)
    res_img.save(filename)
