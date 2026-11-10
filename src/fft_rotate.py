"""Rotate the actin image by 45 degrees and compare the Fourier images, in napari.

Run from the top folder of the repository with:

    uv run src/fft_rotate.py
"""

import napari
import numpy as np
import tifffile
from scipy import ndimage

ANGLE = 45  # degrees

image = tifffile.imread("testImages/Actin_cropped.tif").astype(np.float32)

# reshape=False keeps the output the same size as the input, so the corners
# that rotate out of the picture are cut off, and the new corners are filled with 0
rotated = ndimage.rotate(image, ANGLE, reshape=False)


def log_amplitude(img):
    """log(1 + amplitude) of the Fourier image, with frequency 0 in the center."""
    return np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(img))))


fourier_original = log_amplitude(image)
fourier_rotated = log_amplitude(rotated)

print(f"Image shape: {image.shape}")
print(f"Rotated by {ANGLE} degrees, shape {rotated.shape}")

viewer = napari.Viewer()
viewer.add_image(image, name="Actin_cropped")
viewer.add_image(rotated, name=f"rotated {ANGLE} degrees")
viewer.add_image(fourier_original, name="log amplitude, original")
viewer.add_image(fourier_rotated, name="log amplitude, rotated")
viewer.grid.enabled = True
viewer.grid.shape = (2, 2)  # images on the top row, Fourier images below
napari.run()
