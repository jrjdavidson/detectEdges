# This is python script for Metashape Pro.
#
# Based on https://github.com/agisoft-llc/metashape-scripts/blob/master/src/automatic_masking.py
#
# How to install (Linux):
#
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
#    copy automatic_masking.py script to /home/<username>/.local/share/Agisoft/Metashape Pro/scripts/
# 2. Restart Metashape
#
# How to install (Windows):
#
# 1. Add this script to auto-launch - https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start
#    copy maskingGUI.py script to C:/Users/<username>/AppData/Local/Agisoft/Metashape Pro/scripts/
# 2. Restart Metashape

import Metashape
from PySide2 import QtGui, QtCore, QtWidgets

import time
import tempfile
from pathlib import Path

import urllib.request
from PIL import Image

from cameraMaskingEdges import createMaskCanny, createMaskPillow, setMasks

from modules.pip_auto_install import pip_install

# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(
        found_major_version, compatible_major_version))

try:
    import cv2
    import numpy as np
    import scipy

except ImportError:
    # install dependencies only if import fails to avoid network requests and repetive installations
    temporary_file = tempfile.NamedTemporaryFile(delete=False)
    find_links_file_url = "https://raw.githubusercontent.com/agisoft-llc/metashape-scripts/master/misc/links.txt"
    urllib.request.urlretrieve(find_links_file_url, temporary_file.name)

    pip_install("""-f {find_links_file_path}
scipy == 1.12.0
numpy == 1.26.3
opencv-python == 4.9.0.80""".format(find_links_file_path=temporary_file.name.replace("\\", "\\\\")))


def getCameraImageArray(camera):
    photo_image = camera.photo.image()

    image_types_mapping = {'U8': np.uint8, 'U16': np.uint16}
    if photo_image.data_type not in image_types_mapping:
        print("Image type is not supported yet: {}".format(
            photo_image.data_type))
    if photo_image.cn not in {3, 4}:
        print("Image channels number not supported yet: {}".format(photo_image.cn))
    img = np.frombuffer(photo_image.tostring(), dtype=image_types_mapping[photo_image.data_type]).reshape(
        photo_image.height, photo_image.width, photo_image.cn)[:, :, :3]

    if photo_image.data_type == "U16":
        assert img.dtype == np.uint16
        img = img - np.min(img)
        img = np.float32(img) * 255.0 / np.max(img)
        img = (img + 0.5).astype(np.uint8)
    assert img.dtype == np.uint8

    img = Image.fromarray(img)
    max_downscale = 4
    min_resolution = 640
    downscale = min(photo_image.height // min_resolution,
                    photo_image.width // min_resolution)
    downscale = min(downscale, max_downscale)
    if downscale > 1:
        img = img.resize((photo_image.width // downscale,
                          photo_image.height // downscale))
    img = np.array(img)

    return img


def getPixmapFromArray(imageArray):
    # Convert NumPy array to QImage
    try:
        height, width, channel = imageArray.shape
        imageFormat = QtGui.QImage.Format_RGB888
    except:
        height, width = imageArray.shape
        channel = 1
        imageFormat = QtGui.QImage.Format_Indexed8

    bytes_per_line = channel * width
    qImage = QtGui.QImage(imageArray.data, width, height,
                          bytes_per_line, imageFormat)

    # Convert QImage to QPixmap
    pixmap = QtGui.QPixmap.fromImage(qImage)
    return pixmap


class EdgeMaskDlg(QtWidgets.QDialog):

    def __init__(self, parent):

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Create masks")

        self.edgeModels = []
        self.edgeModels.append(('Canny Edge Detection', 'Canny'))
        self.edgeModels.append(('Pillow Find_edges', 'Pillow'))

        self.modelChoice = QtWidgets.QComboBox()
        for (label, _) in self.edgeModels:
            self.modelChoice.addItem(label)
        (label, _) = self.edgeModels[self.modelChoice.currentIndex()]

        self.chunk = Metashape.app.document.chunk

        self.cameraChoice = QtWidgets.QComboBox()
        if not len(self.chunk.cameras):
            raise Exception("No cameras in chunk")

        for c in self.chunk.cameras:
            self.cameraChoice.addItem(c.label)
        self.cameraChoice.setCurrentIndex(0)
        self.activeCamera = self.chunk.cameras[0]

        self.activeCameraLabel = self.cameraChoice.currentText()
        # self.activeCameraData = getPhoto(self.activeCamera)

        self.noisePixel = QtWidgets.QLabel()
        self.noisePixel.setText("Pixel separation for noise:")
        self.edtNoisePixel = QtWidgets.QLineEdit()
        scale_ratio_tooltip = "Set the minimum distance between noise pixels. All unique isolated pixels that are more distant to another pixel than this value will be masked. If set too high, noise will not be masked. "
        self.noisePixel.setToolTip(scale_ratio_tooltip)
        self.edtNoisePixel.setToolTip(scale_ratio_tooltip)
        noiseValidator = QtGui.QIntValidator()
        noiseValidator.setBottom(0)
        self.edtNoisePixel.setValidator(noiseValidator)

        self.coalesceFactor = QtWidgets.QLabel()
        self.coalesceFactor.setText("Coalesce Island Factor:")
        self.edtCoalesceFactor = QtWidgets.QLineEdit()
        target_resolution_tooltip = "Grow separate \"mask islands\"."
        self.coalesceFactor.setToolTip(target_resolution_tooltip)
        self.edtCoalesceFactor.setToolTip(target_resolution_tooltip)
        cfValdiator = QtGui.QIntValidator()
        cfValdiator.setBottom(0)
        self.edtCoalesceFactor.setValidator(cfValdiator)

        self.threshold = QtWidgets.QLabel()
        self.threshold.setText("Edge Detection threshold:")
        self.edtThreshold = QtWidgets.QLineEdit()
        thresholdTooltip = "A first pass at removing background noise. Value below this value will be ignored (range from 0 to 255)"
        self.threshold.setToolTip(thresholdTooltip)
        self.edtThreshold.setToolTip(thresholdTooltip)
        thresholdValdiator = QtGui.QIntValidator()
        thresholdValdiator.setRange(0, 255)
        self.edtThreshold.setValidator(thresholdValdiator)

        self.threshold1 = QtWidgets.QLabel()
        self.threshold1.setText("Edge Detection threshold:")
        self.edtThreshold1 = QtWidgets.QLineEdit()
        scale_ratio_tooltip = "If a gradient value at a pixel is above this value, it is considered as a potential edge pixel. Default is 50."
        self.threshold1.setToolTip(scale_ratio_tooltip)
        self.edtThreshold1.setToolTip(scale_ratio_tooltip)
        thresholdValdiator1 = QtGui.QIntValidator()
        thresholdValdiator1.setBottom(0)
        self.edtThreshold1.setValidator(thresholdValdiator1)

        self.threshold2 = QtWidgets.QLabel()
        self.threshold2.setText("Edge Detection threshold:")
        self.edtThreshold2 = QtWidgets.QLineEdit()
        scale_ratio_tooltip = "Second threshold for the hysteresis procedure. It is used to decide which edges will be kept. Pixels with gradient values above threshold2 will be considered as edges, and pixels below this threshold will be suppressed. Default is 100"
        self.threshold2.setToolTip(scale_ratio_tooltip)
        self.edtThreshold2.setToolTip(scale_ratio_tooltip)
        thresholdValdiator2 = QtGui.QIntValidator()
        thresholdValdiator2.setBottom(0)
        self.edtThreshold2.setValidator(thresholdValdiator2)

        self.noiseDil = QtWidgets.QLabel()
        self.noiseDil.setText("Noise Dilation:")
        self.edtNoiseDil = QtWidgets.QLineEdit()
        scale_ratio_tooltip = 'Dilation filter iteration number. Can be used in conjunction with "Noise Erosion" to attempt to remove noise.'
        self.noiseDil.setToolTip(scale_ratio_tooltip)
        self.edtNoiseDil.setToolTip(scale_ratio_tooltip)
        noiseDilValidator = QtGui.QIntValidator()
        noiseDilValidator.setBottom(0)
        self.edtNoiseDil.setValidator(noiseDilValidator)

        self.noiseEro = QtWidgets.QLabel()
        self.noiseEro.setText("Noise Erosion:")
        self.edtNoiseEro = QtWidgets.QLineEdit()
        scale_ratio_tooltip = 'Erosion filter iteration number Can be used in conjunction with "Noise Dilation" to attempt to remove noise.'
        self.noiseEro.setToolTip(scale_ratio_tooltip)
        self.edtNoiseEro.setToolTip(scale_ratio_tooltip)
        noiseErosionValidator = QtGui.QIntValidator()
        noiseErosionValidator.setBottom(0)
        self.edtNoiseEro.setValidator(noiseErosionValidator)

        self.coaDil = QtWidgets.QLabel()
        self.coaDil.setText("Coalescing factor:")
        self.edtCoaDil = QtWidgets.QLineEdit()
        scale_ratio_tooltip = 'Number of dilation and erosion filter iterations. This is typically use to bridge gaps between "mask islands".'
        self.coaDil.setToolTip(scale_ratio_tooltip)
        self.edtCoaDil.setToolTip(scale_ratio_tooltip)
        coaDilValidator = QtGui.QIntValidator()
        coaDilValidator.setBottom(0)
        self.edtCoaDil.setValidator(coaDilValidator)

        self.kernelSize = QtWidgets.QLabel()
        self.kernelSize.setText("Kernel size:")
        self.edtKernelSize = QtWidgets.QLineEdit()
        scale_ratio_tooltip = "Kernel size for Coalescing pass. Default is 5 (a 5x5 kernel) "
        self.kernelSize.setToolTip(scale_ratio_tooltip)
        self.edtKernelSize.setToolTip(scale_ratio_tooltip)
        kernelSizeValidator = QtGui.QIntValidator()
        kernelSizeValidator.setBottom(0)
        self.edtKernelSize.setValidator(kernelSizeValidator)

        cannyArgs = self.getCannyArgs()
        pillowArgs = self.getPillowArgs()

        self.edtNoisePixel.setPlaceholderText(str(pillowArgs['noisePixelSeperation']))
        self.edtCoalesceFactor.setPlaceholderText(str(pillowArgs['coalesceFactor']))
        self.edtThreshold.setPlaceholderText(str(pillowArgs['threshold']))
        self.edtThreshold1.setPlaceholderText(str(cannyArgs['threshold1']))
        self.edtThreshold2.setPlaceholderText(str(cannyArgs['threshold2']))
        self.edtNoiseDil.setPlaceholderText(str(cannyArgs['noiseDilation']))
        self.edtNoiseEro.setPlaceholderText(str(cannyArgs['noiseErosion']))
        self.edtCoaDil.setPlaceholderText(str(cannyArgs['coalescingfactor']))
        self.edtKernelSize.setPlaceholderText(str(cannyArgs['kernelsize']))


        # Set the image to the QLabel
        self.imageEdgeLabel = QtWidgets.QLabel()
        self.imageNoiseLabel = QtWidgets.QLabel()
        self.imageFinalLabel = QtWidgets.QLabel()

        self.updateCameraLabels()

        self.btnOk = QtWidgets.QPushButton("Generate All Masks")
        self.btnOk.setFixedSize(150, 30)
        self.btnOk.setToolTip("Start Script to generate all masks.")

        self.btnFullSize = QtWidgets.QPushButton("Show Full Size Overlay")
        self.btnFullSize.setFixedSize(150, 30)
        self.btnFullSize.setToolTip(
            "Open a new window to show full-size overlay image")

        self.btnTest = QtWidgets.QPushButton("Test")
        self.btnTest.setFixedSize(150, 30)
        self.btnTest.setToolTip(
            "Test the current setting with selected photo.")

        self.btnQuit = QtWidgets.QPushButton("Close")
        self.btnQuit.setFixedSize(150, 30)

        layout = QtWidgets.QGridLayout()
        self.cannyElements = [
            self.threshold1,
            self.edtThreshold1,
            self.threshold2,
            self.edtThreshold2,
            self.noiseDil,
            self.edtNoiseDil,
            self.noiseEro,
            self.edtNoiseEro,
            self.coaDil,
            self.edtCoaDil,
            self.kernelSize,
            self.edtKernelSize,
        ]
        self.pillowElements = [
            self.noisePixel,
            self.edtNoisePixel,
            self.coalesceFactor,
            self.edtCoalesceFactor,
            self.threshold,
            self.edtThreshold]

        layout.addWidget(self.cameraChoice, 0, 0)
        layout.addWidget(self.modelChoice, 0, 1)

        # Pillow widgets

        layout.addWidget(self.threshold, 1, 0)
        layout.addWidget(self.edtThreshold, 1, 1)

        layout.addWidget(self.noisePixel, 1, 2)
        layout.addWidget(self.edtNoisePixel, 1, 3)

        layout.addWidget(self.coalesceFactor, 1, 4)
        layout.addWidget(self.edtCoalesceFactor, 1, 5)

        # CAnny widgets

        layout.addWidget(self.threshold1, 1, 0)
        layout.addWidget(self.edtThreshold1, 1, 1)
        layout.addWidget(self.threshold2, 2, 0)
        layout.addWidget(self.edtThreshold2, 2, 1)

        layout.addWidget(self.noiseDil, 1, 2)
        layout.addWidget(self.edtNoiseDil, 1, 3)
        layout.addWidget(self.noiseEro, 2, 2)
        layout.addWidget(self.edtNoiseEro, 2, 3)

        layout.addWidget(self.coaDil, 1, 4)
        layout.addWidget(self.edtCoaDil, 1, 5)

        layout.addWidget(self.kernelSize, 2, 4)
        layout.addWidget(self.edtKernelSize, 2, 5)

        layout.addWidget(self.imageEdgeLabel, 3, 0, 1, 2)
        layout.addWidget(self.imageNoiseLabel, 3, 2, 1, 2)
        layout.addWidget(self.imageFinalLabel, 3, 4, 1, 2)

        layout.addWidget(self.btnOk, 5, 0)
        layout.addWidget(self.btnTest, 5, 1)
        layout.addWidget(self.btnFullSize, 5, 2)
        layout.addWidget(self.btnQuit, 5, 3)

        self.setLayout(layout)

        self.updateModelChoice()

        self.overlayedImage = None

        self.btnOk.clicked.connect(self.confirm)
        self.btnTest.clicked.connect(self.test)
        self.btnQuit.clicked.connect(self.reject)
        self.btnFullSize.clicked.connect(self.createFullSize)
        self.cameraChoice.currentIndexChanged.connect(self.updateCameraLabels)
        self.modelChoice.currentIndexChanged.connect(self.updateModelChoice)

        self.exec()

    def updateModelChoice(self):

        (_, modelType) = self.edgeModels[self.modelChoice.currentIndex()]

        if (modelType == "Canny"):
            for element in self.pillowElements:
                element.hide()
            for element in self.cannyElements:
                element.show()
        elif (modelType == "Pillow"):
            for element in self.pillowElements:
                element.show()
            for element in self.cannyElements:
                element.hide()
        else:
            raise Exception(f"Undefined model type : {modelType}")

    def updateCameraLabels(self):

        self.activeCamera = self.chunk.cameras[
            self.cameraChoice.currentIndex()]

        self.activeCameraPixmap = getPixmapFromArray(
            getCameraImageArray(self.activeCamera))
        self.overlayedImage = None

        for label in [
                self.imageEdgeLabel,
                self.imageNoiseLabel,
                self.imageFinalLabel,
        ]:

            label.setPixmap(self.activeCameraPixmap.scaled(
                label.size(), QtCore.Qt.KeepAspectRatio))

    def createFullSize(self):
        print(self.overlayedImage)
        if self.overlayedImage is None:
            return

        full_size_dialog = QtWidgets.QDialog(self)
        full_size_dialog.setWindowTitle("Full Size Overlay Image")

        full_size_label = QtWidgets.QLabel(full_size_dialog)

        # Get the pixmap from the existing label
        full_size_pixmap = getPixmapFromArray(self.overlayedImage)

        full_size_label.setPixmap(full_size_pixmap)
        full_size_dialog.resize(1200, 1200 / full_size_pixmap.width() * full_size_pixmap.height())

        def resize_label():
            # Get the size of the dialog
            dialog_size = full_size_dialog.size()

            # Get the original size of the pixmap
            original_size = full_size_pixmap.size()

            # Calculate the new dimensions while maintaining the aspect ratio
            aspect_ratio = original_size.width() / original_size.height()

            new_width = dialog_size.width()
            new_height = int(new_width / aspect_ratio)

            # Check if the calculated height exceeds the dialog height
            if new_height > dialog_size.height():
                new_height = dialog_size.height()
                new_width = int(new_height * aspect_ratio)

            # Resize the pixmap
            scaled_pixmap = full_size_pixmap.scaled(new_width, new_height)

            full_size_label.setPixmap(scaled_pixmap)
            full_size_label.resize(scaled_pixmap.width(), scaled_pixmap.height())

        # Connect the resize event to the resize_label method
        full_size_dialog.resizeEvent = lambda event: resize_label()

        resize_label()  # Initially set the label size

        full_size_dialog.exec_()

    def confirm(self):
        print("Script started...")

        (_, modelType) = self.edgeModels[self.modelChoice.currentIndex()]
        if (modelType == "Canny"):

            args = self.getCannyArgs()
        elif (modelType == "Pillow"):

            args = self.getPillowArgs()
        else:
            raise Exception(f"Undefined model type : {modelType}")

        qm = QtWidgets.QMessageBox
        print(args)
        ret = qm.question(
            self, '', "Are you sure? Generating masks can take a few minutes.", qm.Yes | qm.No)

        if ret == qm.Yes:

            try:
                message = setMasks(self.chunk, modelType, **args)
            except:
                message = 'There was an error, please see the console.'
            qm.information(self, '', message)
        else:
            qm.information(self, '', "Nothing Changed")

        self.reject()

    def getCannyArgs(self):
        args = {
            'threshold1': int(self.edtThreshold1.text()) if self.edtThreshold1.text() else 100,
            'threshold2': int(self.edtThreshold2.text()) if self.edtThreshold2.text() else 200,
            'noiseDilation': int(self.edtNoiseDil.text()) if self.edtNoiseDil.text() else 5,
            'noiseErosion': int(self.edtNoiseEro.text()) if self.edtNoiseEro.text() else 15,
            'coalescingfactor': int(self.edtCoaDil.text()) if self.edtCoaDil.text() else 40,
            'kernelsize': int(self.edtKernelSize.text()) if self.edtKernelSize.text() else 5,
        }
        return args

    def getPillowArgs(self):
        args = {
            'threshold': int(self.edtThreshold.text()) if self.edtThreshold.text() else 180,
            'noisePixelSeperation': int(self.edtNoisePixel.text()) if self.edtNoisePixel.text() else 20,
            'coalesceFactor': int(self.edtCoalesceFactor.text()) if self.edtCoalesceFactor.text() else 60,
        }
        return args

    def test(self):
        print("Test script started...")
        (label, modelType) = self.edgeModels[self.modelChoice.currentIndex()]

        currentCamera = self.activeCamera

        if (modelType == "Canny"):

            cannyArgs = self.getCannyArgs()
            edgesImage, noiseImage, finalImage, self.overlayedImage = createMaskCanny(
                currentCamera.photo.path, None, **cannyArgs)
        elif (modelType == "Pillow"):

            pillowArgs = self.getPillowArgs()

            edgesImage, noiseImage, finalImage, self.overlayedImage = createMaskPillow(
                currentCamera.photo.path, None, **pillowArgs)
        else:
            raise Exception(f"Undefined model type : {modelType}")

        for image, label in [(edgesImage, self.imageEdgeLabel), (noiseImage, self.imageNoiseLabel), (finalImage, self.imageFinalLabel)]:
            pixmap = getPixmapFromArray(image)
            label.setPixmap(pixmap.scaled(
                label.size(), QtCore.Qt.KeepAspectRatio))


def show_alignment_dialog():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    dlg = EdgeMaskDlg(parent)


label = "Scripts/Create masks for out of focus photos"
Metashape.app.addMenuItem(label, show_alignment_dialog)
print("To execute this script press {}".format(label))
