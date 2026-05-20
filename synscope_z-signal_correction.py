import os
import sys

from pathlib import Path,PurePath
sys.path.append('../')

import cv2
import tifffile
import argparse
import numpy as np

import tensorflow as tf
import tensorflow_addons as tf

from PIL import Image
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.ISCL.utils.image_tool import *
from utils.ISCL.utils.parser import parse_args
from utils.ISCL.utils.callbacks import Val_Init
from utils.ISCL.models.trainer import Trainer


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ('yes', 'true', 't', 'y', '1'):
        return True
    if value in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}')


def infer_output_dtype(frame: np.ndarray) -> str:
    if frame.dtype == np.uint8:
        return 'uint8'
    if frame.dtype == np.uint16:
        return 'uint16'
    if np.issubdtype(frame.dtype, np.integer):
        return 'uint8' if frame.max() <= np.iinfo(np.uint8).max else 'uint16'
    if np.issubdtype(frame.dtype, np.floating):
        return 'uint8' if frame.max() <= np.iinfo(np.uint8).max else 'uint16'
    return 'uint16'

def ISCL(args, train_clean, train_noisy, simulated_data):

    '''
    reference: Lee et al. "ISCL: Interdependent Self-Cooperative Learning for Unpaired Image Denoising" IEEE Transactions on Medical Imaging, (2021)

    '''

    clean = image_division(train_clean, patch_size=(64,64))
    noisy = image_division(train_noisy, patch_size=(64,64))

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE = args.batch_size*strategy.num_replicas_in_sync
    # Preprocessing
    dataset = tf.data.Dataset.from_tensor_slices((clean, noisy))
    dataset = dataset.cache().repeat().shuffle(len(train_clean), reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    print(args.training)
    if args.training:
        # Training
        with strategy.scope():
            model = Trainer(args)
            model.compile()
        callbacks = []
        model.fit(dataset, epochs=args.epoch, callbacks=callbacks, steps_per_epoch=args.iter)
        model._save(args.result_dir+"/model/my_model")
    else:
        model = Trainer(args)
        model._load(args.result_dir+"/model/my_model")

    # Testing
    pred = np.zeros(np.shape(simulated_data), dtype=np.float32)

    for i in range(0,len(simulated_data)):
        temp = np.squeeze(model.predict(simulated_data[i:i+1,:,:,np.newaxis],ensemble=True)) # noisy: [1, H, W, 1] or [1, H, W, C]
        pred[i:i+1] = temp

    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=400)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clean_slide', type=int, nargs='+')
    parser.add_argument('--noisy_slide', type=int, nargs='+')
    parser.add_argument('--target_range', type=int, nargs='+')
    parser.add_argument('--ref_slide', type=int, default=3)
    parser.add_argument('--clip_limit', type=float, default=1.5)
    parser.add_argument('--training', type=str2bool, default=False)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--dtype', type=str, default=None, choices=['uint8', 'uint16'])

    # Load experiment setting
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "model"), exist_ok=True)

    path = Image.open(args.data)
    data = []
    n = path.n_frames

    for i in range(n):
        path.seek(i)
        temp_original = np.array(path)

        # Check the original data type and convert appropriately
        if temp_original.dtype == np.uint16:
            # For uint16 images, normalize to uint8 range
            temp = (255 * (temp_original / temp_original.max())).astype(np.uint8)
        elif temp_original.dtype == np.uint8:
            # For uint8 images, use directly
            temp = temp_original.copy()
        else:
            # For other types, convert to uint8
            temp = temp_original.astype(np.uint8)

        if i == 0:
            x, y = temp.shape
            inferred_dtype = infer_output_dtype(temp_original)
            if args.dtype is None:
                args.dtype = inferred_dtype
            print("Data shape of an original input", (n, x, y))
            print(f"Original data type: {temp_original.dtype}, Converted to: {temp.dtype}")
            print(f"Output data type: {args.dtype}")

        temp = np.pad(temp, ((0, 4 - x % 4), (0, 4 - y % 4)), 'constant', constant_values=0)
        data.append(temp)

    data = np.array(data, dtype=np.uint8)
    print("Data shape of a preprocessed input", data.shape)

    clahe = cv2.createCLAHE(clipLimit=args.clip_limit, tileGridSize=(64, 64))
    data_enhance = []
    reference_slide = np.array(data[args.ref_slide], dtype=np.uint8)
    reference = clahe.apply(reference_slide)

    [_, fname] = os.path.split(args.data)
    fname = os.path.splitext(fname)[0]

    for i in tqdm(range(n), desc="Image Enhancement"):
        if args.target_range[0] < i < args.target_range[1]:
            temp = clahe.apply(data[i])
            data_enhance.append(match_histograms(temp, reference))
        else:
            data_enhance.append(data[i])

    data_enhance = np.array(data_enhance, dtype=np.float32)
    data_enhance = (data_enhance / 255) * 2 - 1

    clean = np.array(data_enhance[args.clean_slide], dtype=np.float32)
    noisy = np.array(data_enhance[args.noisy_slide], dtype=np.float32)

    output = ISCL(args, clean, noisy, data_enhance)
    output = output[:, :x, :y]

    if args.dtype == 'uint16':
        # Scale from [0, 255] to [0, 65535]
        output_uint = (output.astype(np.float32) * (65535.0 / 255.0))
        output_uint = np.clip(output_uint, 0, 65535).astype(np.uint16)
    else:
        # For uint8, just clip and cast
        output_uint = np.clip(output, 0, 255).astype(np.uint8)

    tifffile.imwrite(os.path.join(args.result_dir, fname + "_denoised_image.tiff"), output_uint)

if __name__ == '__main__':
    main()
