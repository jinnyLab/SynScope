import os
import sys

sys.path.append('../')

import csv

from zimg import *
from utils.logger import setup_logger
logger = setup_logger()

def export_puncta_info(img_folder: str):

    print('Processing puncta...')

    (_, _, file_list) = next(os.walk(os.path.join(img_folder)))
    puncta_list = [fn for fn in file_list if '.nimp' in fn]
    nimgs = len(puncta_list)

    for puncta_idx in range(nimgs):
        print(puncta_list[puncta_idx])

        puncta = ZPuncta(os.path.join(img_folder, puncta_list[puncta_idx]))
        puncta_loc_list = puncta.data

        num_puncta = len(puncta_loc_list)
        puncta_name = puncta_list[puncta_idx].split('.')[0]
        output_csv = os.path.join(img_folder, f'{puncta_name}_puncta_info.csv')

        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(
                ['puncta #', 'score', 'punctum x', 'punctum y', 'punctum z', 'radius', 'volume', 'mass', 'mean intensity',
                 'max intensity', 'intensity SD'])

            for pi, punctum in enumerate(puncta_loc_list):
                logger.info(f'{pi} / {num_puncta}')

                pi = pi + 1
                score = punctum.score
                punctum_x = punctum.x
                punctum_y = punctum.y
                punctum_z = punctum.y
                radius = punctum.radius
                volume = punctum.volSize
                mass = punctum.mass
                mean_intensity = punctum.meanIntensity
                max_intensity = punctum.maxIntensity
                intensity_SD = punctum.sDevOfIntensity

                writer.writerow([pi, score, punctum_x, punctum_y, punctum_z, radius, volume, mass, mean_intensity,
                                 max_intensity, intensity_SD,])

if __name__ == '__main__':

    img_folder = 'path/to/puncta/folder'

    export_puncta_info(img_folder)
