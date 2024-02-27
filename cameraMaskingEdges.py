import Metashape
import pathlib
import scipy
import cv2
# import concurrent.futures
import numpy as np
from PIL import Image, ImageFilter


def getCameras(chunk=None):

    print("Script started...")
    if chunk is None:
        chunk = Metashape.app.document.chunk

    cameras = chunk.cameras

    nmasks_exists = 0
    for c in cameras:
        if c.mask is not None:
            nmasks_exists += 1
            print("Camera {} already has mask".format(c.label))
    if nmasks_exists > 0:
        raise Exception(
            "There are already {} masks, please remove them and try again".format(nmasks_exists))

    masks_dirs_created = set()
    cameras_by_masks_dir = {}
    for _, c in enumerate(cameras):
        if not c.type == Metashape.Camera.Type.Regular:  # skip camera track, if any
            continue

        input_image_path = c.photo.path
        image_mask_dir = pathlib.Path(input_image_path).parent / 'masks'
        if image_mask_dir.exists() and str(image_mask_dir) not in masks_dirs_created:
            attempt = 2
            image_mask_dir_attempt = pathlib.Path(
                str(image_mask_dir) + "_{}".format(attempt))
            while image_mask_dir_attempt.exists() and str(image_mask_dir_attempt) not in masks_dirs_created:
                attempt += 1
                image_mask_dir_attempt = pathlib.Path(
                    str(image_mask_dir) + "_{}".format(attempt))
            image_mask_dir = image_mask_dir_attempt
        if image_mask_dir.exists():
            assert str(image_mask_dir) in masks_dirs_created
        else:
            image_mask_dir.mkdir(parents=False, exist_ok=False)
            masks_dirs_created.add(str(image_mask_dir))
            cameras_by_masks_dir[str(image_mask_dir)] = list()
        cameras_by_masks_dir[str(image_mask_dir)].append(c)

    return chunk, cameras, cameras_by_masks_dir, masks_dirs_created


def processCamera(image_mask_dir, c, camera_index, cameras):

    # skip camera track, if any
    if not c.type == Metashape.Camera.Type.Regular:
        return

    input_image_path = c.photo.path
    print("{}/{} processing: {}".format(camera_index + 1,
                                        len(cameras), input_image_path))

    image_mask_name = pathlib.Path(input_image_path).name.split(".")
    if len(image_mask_name) > 1:
        image_mask_name = image_mask_name[:-1]

    image_mask_name = ".".join(image_mask_name)

    image_mask_path = str(image_mask_dir / image_mask_name) + "_mask.png"

    return input_image_path, image_mask_path


def createMaskPillow(imagePath, maskPath=None, threshold=180, noisePixelSeperation=20, coalesceFactor=120):
    with Image.open(imagePath) as image:
        # Make greyscale copy
        grey = image.copy().convert('L')
        # Find the edges
        edges = grey.filter(ImageFilter.FIND_EDGES)
        print('Filtering edges below threshold')

        # Remove any value below threshold( removes darker pixels )
        edges = edges.point(lambda x: 255 if x > threshold else 0)

        # convert to binary image (i.e. 0 or 1 for each pixel)
        print("Removing noise.")
        mask = edges.convert("1")
        mask = np.array(mask)
        noiseCycles = int(noisePixelSeperation / 2)
        maskNoise = scipy.ndimage.binary_dilation(mask, iterations=noiseCycles)
        maskNoise = scipy.ndimage.binary_erosion(
            maskNoise, iterations=noiseCycles + 1)
        Metashape.app.update()
        print("Coalescing separate edges.")
        maskFinal = scipy.ndimage.binary_dilation(
            maskNoise, iterations=coalesceFactor)
        maskFinal = scipy.ndimage.binary_erosion(
            maskFinal, iterations=coalesceFactor)
        Metashape.app.update()
        maskFinal = maskFinal.astype(np.uint8) * 255
        maskFinal = np.dstack([maskFinal, maskFinal, maskFinal])
        Metashape.app.update()

        if (maskPath):
            Image.fromarray(maskFinal).save(maskPath)
            print(f'Created mask from file {image.filename}')
        else:

            # Overlay the final mask on the original image using NumPy array operations
            overlayed_image = np.where(maskFinal == 0, maskFinal, np.array(image))

            maskNoise = maskNoise.astype(np.uint8) * 255
            maskNoise = np.dstack([maskNoise, maskNoise, maskNoise])
            return mask, maskNoise, maskFinal, overlayed_image


def createMaskCanny(imagePath, maskPath=None, threshold1=100, threshold2=200, noiseDilation=5, noiseErosion=15, coalescingfactor=40, kernelsize=5):

    # Read the image
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    # Apply Canny edge detection
    edges = cv2.Canny(image, threshold1, threshold2)

    print('Filtering edges below threshold')
    # Apply thresholding
    _, binary_edges = cv2.threshold(edges, 180, 255, cv2.THRESH_BINARY)
    Metashape.app.update()

    # Invert the binary image
    # inverted_mask = cv2.bitwise_not(binary_edges)

    # Apply the inverted mask to the original image
    # Apply binary dilation
    # Adjust the kernel size as needed
    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    dilated_mask1 = cv2.dilate(binary_edges, kernel, iterations=noiseDilation)
    eroded_mask1 = cv2.erode(dilated_mask1, kernel, iterations=noiseErosion)
    dilated_mask2 = cv2.dilate(
        eroded_mask1, kernel, iterations=coalescingfactor)
    eroded_mask2 = cv2.erode(dilated_mask2, kernel,
                             iterations=coalescingfactor)
    Metashape.app.update()

    # Apply the dilated mask to the original image

    if (maskPath):
        cv2.imwrite(maskPath, eroded_mask2)
        print(f'Created mask from file {imagePath}')

    else:
        masked_image = cv2.bitwise_and(image, image, mask=eroded_mask2)
        return binary_edges, eroded_mask1, eroded_mask2, masked_image


def setMasks(chunk, type, **kwargs):

    chunk, cameras, cameras_by_masks_dir, masks_dirs_created = getCameras(
        chunk)

    camera_offset = 0
    print(kwargs)
    for masks_dir, dir_cameras in cameras_by_masks_dir.items():
        for camera_index, c in enumerate(dir_cameras):
            image, maskPath = processCamera(pathlib.Path(
                masks_dir), c, camera_offset + camera_index, cameras)
            if (type == "Canny"):
                createMaskCanny(image, maskPath, **kwargs)
            elif (type == "Pillow"):
                createMaskPillow(image, maskPath, **kwargs)
            else:
                raise Exception(f"Undefined Type: {type}")

        camera_offset += len(dir_cameras)

    print("{} masks generated in {} directories:".format(
        len(cameras), len(masks_dirs_created)))
    for mask_dir in sorted(masks_dirs_created):
        print(mask_dir)

    print("Importing masks into project...")
    for masks_dir, dir_cameras in cameras_by_masks_dir.items():
        chunk.generateMasks(path=masks_dir + "/{filename}_mask.png",
                            masking_mode=Metashape.MaskingMode.MaskingModeFile,
                            cameras=dir_cameras)

    print("Script finished.")

    return "{} masks generated in {} directories. Masks were added to photos.".format(
        len(cameras), len(masks_dirs_created))
