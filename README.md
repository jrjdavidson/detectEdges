# Detect edges and create mask in metashape

This repor has two python scripts:
- one script that can be used to mask photos that are out of focus.
- one script adds a GUI to Agisoft Metashape to run the aforementioned script. 

I created it as I felt this functionality should exist in metashape! Uses [ImageFilter.FIND_EDGES from Pillow](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html) or [binary_edges from the OpenCV threshold function](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) to detect edges. Next, noise is removed and areas are consolidated by using binary dilation and erosion filters. 

I've provided images for testing purposes, and they also illustrate cases where the filter should work well. All photos of objects that are highly textured should work well. ![example](/example.png "GUI photo").