import os
import sys

from pathlib import Path,PurePath
sys.path.append('../')

from zimg import *

def run_puncta_detection(
    image_folder: str,
    filename: str,
    result_folder: str = None,
    mGRASP_channel: int = 3,
    dendrite_channel: int = 1,
    threshold: int = -1,
    voxelSize_X: float = 0.0,
    voxelSize_Y: float = 0.0,
    voxelSize_Z: float = 0.0,
    swc_name: str | None = None,
):
    '''
    reference: Feng et al. "Improved synapse detection for mGRASP-assisted brain connectivity mapping" Bioinformatics, 28(2012)

    '''
    if not result_folder:
        result_folder = os.path.join(image_folder)
    os.makedirs(result_folder, exist_ok=True)

    log_dir = os.path.join(result_folder, "log")
    os.makedirs(log_dir, exist_ok=True)

    voxelSize_X = voxelSize_X
    voxelSize_Y = voxelSize_Y
    voxelSize_Z = voxelSize_Z

    punctaDetection = ZPunctaDetection()

    if mGRASP_channel <= 0 or dendrite_channel <= 0:
        raise ValueError("mGRASP_channel and dendrite_channel must be >= 1 (1-based indexing).")

    punctaChannel = mGRASP_channel - 1
    dendriteChannel = dendrite_channel - 1
    t = 0
    scene = 0
    voxelSizeInUmX = voxelSize_X
    voxelSizeInUmY = voxelSize_Y
    voxelSizeInUmZ = voxelSize_Z

    # Optional SWC file for dendrite morphology; only set if provided
    if swc_name is not None:
        swc_filename = os.path.join(image_folder, swc_name)

    punctaDetection.setPunctaThreshold(threshold)
    punctaDetection.setSomaPunctaThreshold(-1)
    punctaDetection.setSplitThreshold(20)
    punctaDetection.setConfidenceRegionForRadiusEstimate(9.5e-1)
    punctaDetection.setConfidenceRegionForOverlapArea(8e-1)
    punctaDetection.setOverlapRateThreshold(8e-1)
    punctaDetection.setSeedSizeThreshold(6)
    punctaDetection.setUseMultithreading(True)
    punctaDetection.setDendriteChannel(dendriteChannel)
    punctaDetection.setMaxDendriteTubeRadiusInUm(2.6e0)
    punctaDetection.setDendriteThreshold(1e2)
    punctaDetection.setMaxDistToBranchInUm(2.5e0)
    punctaDetection.setAmbiguousFactor(1e0)

    # Only configure SWC files if an SWC was supplied
    if swc_name is not None:
        punctaDetection.setSwcFiles([swc_filename])

    print("Processing", filename)

    input_filename = os.path.join(image_folder, filename)

    if swc_name is not None:
        swc_stem = PurePath(swc_name).stem
        punctaFilename = os.path.join(result_folder, f"{swc_stem}_puncta.nimp")
        somaPunctaFilename = os.path.join(result_folder, f"{swc_stem}_soma_puncta.nimp")
    else:
        punctaFilename = os.path.join(
            result_folder, f"{PurePath(filename).stem}_detected_puncta.nimp"
        )
        somaPunctaFilename = os.path.join(
            result_folder, f"{PurePath(filename).stem}_detected_soma_puncta.nimp"
        )

    punctaDetection.setLogFile(
        os.path.join(log_dir, f"{PurePath(filename).stem}_puncta_detection_log.txt")
    )
    punctaDetection.setInputFile(
        input_filename,
        punctaChannel,
        t,
        scene,
        voxelSizeInUmX,
        voxelSizeInUmY,
        voxelSizeInUmZ,
    )

    punctaDetection.setResultPunctaFilename(punctaFilename)
    punctaDetection.setResultSomaPunctaFilename(somaPunctaFilename)

    punctaDetection.run()


if __name__ == "__main__":

    image_folder = 'path/to/image/folder'
    filename = 'image_name.tiff'

    swc_name = None # set to None if you don't want to use an SWC

    threshold = -1
    mGRASP_channel = 4
    dendrite_channel = 2
    voxel_X = 0.23
    voxel_Y = 0.23
    voxel_Z = 0.5

    run_puncta_detection(
        image_folder,
        filename,
        result_folder=None,
        mGRASP_channel=mGRASP_channel,
        dendrite_channel=dendrite_channel,
        threshold=threshold,
        swc_name=swc_name,
        voxelSize_X=voxel_X,
        voxelSize_Y=voxel_Y,
        voxelSize_Z=voxel_Z
    )
