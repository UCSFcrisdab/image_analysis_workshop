"""Find edges in the actin image with the Sobel x and y kernels.

Run from the top folder of the repository with:

    uv run src/sobel.py

Results are saved in Session3_Filters/Cristina_Session3/output/ as
sobel_x.tif, sobel_y.tif, sobel_mag.tif and sobel_mag_blurred.tif (float32).
sobel_mag_blurred.tif is the gradient magnitude after first blurring with a 5 x 5
Gaussian.  laplacian.tif is the input convolved with the 3 x 3 Laplacian kernel.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

input_path = "testImages/Actin_cropped.tif"
out_dir = Path("Session3_Filters/Cristina_Session3/output")

# Right column minus left column: positive where the image gets brighter to the right
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
# Bottom row minus top row: positive where the image gets brighter going down
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

# Read the image as decimal numbers (float) and its pixel size (pixels per micron)
with tifffile.TiffFile(input_path) as tif:
    image = tif.asarray().astype(np.float32)
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")
pixel_size = denominator / numerator

def sobel(img):
    # correlate applies the kernel as written; convolve would flip it and swap the signs
    sx = ndimage.correlate(img, sobel_x, mode="reflect")
    sy = ndimage.correlate(img, sobel_y, mode="reflect")
    # Edge strength regardless of direction (Pythagoras on the x and y slopes)
    return sx, sy, np.sqrt(sx**2 + sy**2)


sx, sy, magnitude = sobel(image)

# Blur first with a 5 x 5 Gaussian (sigma 1, radius 2) so noise makes fewer false edges
blurred = ndimage.gaussian_filter(image, sigma=1.0, radius=2, mode="reflect")
_, _, magnitude_blurred = sobel(blurred)

# 8 x center minus its 8 neighbors: the weights add up to 0, so flat areas give 0.
# Positive where a pixel is brighter than its surroundings, negative where darker.
laplacian_kernel = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]], dtype=np.float32)
laplacian = ndimage.convolve(image, laplacian_kernel, mode="reflect")

print(f"Input: {input_path}, min {image.min():.1f}, max {image.max():.1f}, "
      f"pixel size {pixel_size:.4f} {unit}")
out_dir.mkdir(parents=True, exist_ok=True)
for filename, data in [("sobel_x.tif", sx), ("sobel_y.tif", sy), ("sobel_mag.tif", magnitude),
                       ("sobel_mag_blurred.tif", magnitude_blurred), ("laplacian.tif", laplacian)]:
    # Store the pixel size again, so Fiji and napari show the right scale
    tifffile.imwrite(out_dir / filename, data.astype(np.float32), imagej=True,
                     resolution=(1 / pixel_size, 1 / pixel_size),
                     metadata={"unit": unit})
    print(f"Saved {filename:<22} min {data.min():9.1f}   max {data.max():9.1f}")
