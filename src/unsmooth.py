"""Smooth the noisy nucleus image, then try to undo the smoothing in Fourier space.

Run from the top folder of the repository with:

    uv run src/unsmooth.py

The recovered image is saved as
Session4_FourierAndNoise/Cristina_Session4/output/unsmooth.tif (float32).
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

input_path = "testImages/Nucleus_cropped_noise.tif"
out_path = Path("Session4_FourierAndNoise/Cristina_Session4/output/unsmooth.tif")

KERNEL = np.ones((3, 3)) / 9  # the Smooth kernel from Session 3

# Read the image as decimal numbers (float) and its pixel size (pixels per micron)
with tifffile.TiffFile(input_path) as tif:
    image = tif.asarray().astype(np.float64)
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")
pixel_size = denominator / numerator

# Smooth.  mode="wrap" treats the image as repeating tiles, like the Fourier transform does.
smoothed = ndimage.convolve(image, KERNEL, mode="wrap")

# Put the kernel in an image of the same size, with its center pixel at (0, 0)
kernel_image = np.zeros(image.shape)
kernel_image[:3, :3] = KERNEL
kernel_image = np.roll(kernel_image, (-1, -1), axis=(0, 1))

# Convolution = multiplication in Fourier space, so to undo it, divide
smoothed_f = np.fft.fft2(smoothed)
kernel_f = np.fft.fft2(kernel_image)
with np.errstate(divide="ignore", invalid="ignore"):
    recovered = np.real(np.fft.ifft2(smoothed_f / kernel_f))

print(f"Input:     min {image.min():.1f}, max {image.max():.1f}")
print(f"Smoothed:  min {smoothed.min():.1f}, max {smoothed.max():.1f}")
print(f"Recovered: min {np.nanmin(recovered):.4g}, max {np.nanmax(recovered):.4g}")
print(f"Pixels that are not a number or infinite: {(~np.isfinite(recovered)).sum()}")
print(f"Smallest amplitude of the kernel's Fourier image: {np.abs(kernel_f).min():.2g} "
      f"({(np.abs(kernel_f) < 1e-6).sum()} of {kernel_f.size} values are below 1e-6)")

# Store the pixel size again, so Fiji and napari show the right scale
out_path.parent.mkdir(parents=True, exist_ok=True)
tifffile.imwrite(out_path, recovered.astype(np.float32), imagej=True,
                 resolution=(1 / pixel_size, 1 / pixel_size),
                 metadata={"unit": unit})
print(f"Saved {out_path}")
