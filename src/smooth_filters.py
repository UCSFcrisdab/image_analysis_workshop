"""Smooth a noisy nucleus image by convolving it with three averaging kernels.

Run from the top folder of the repository with:

    uv run src/smooth_filters.py

Results are saved in Session3_Filters/Cristina_Session3/output/ as
smooth_3x3.tif, smooth_3x3_center4.tif and smooth_7x7.tif.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

input_path = "testImages/Nucleus_cropped_noise.tif"
out_dir = Path("Session3_Filters/Cristina_Session3/output")

center4 = np.ones((3, 3))
center4[1, 1] = 4

kernels = {
    "smooth_3x3.tif": np.ones((3, 3)),
    "smooth_3x3_center4.tif": center4,
    "smooth_7x7.tif": np.ones((7, 7)),
}

# Read the image as decimal numbers (float) and its pixel size (pixels per micron)
with tifffile.TiffFile(input_path) as tif:
    image = tif.asarray().astype(np.float32)
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")
pixel_size = denominator / numerator

print(f"Input:  {input_path}, shape {image.shape}, type {image.dtype}, "
      f"pixel size {pixel_size:.4f} {unit}")

out_dir.mkdir(parents=True, exist_ok=True)
for filename, kernel in kernels.items():
    # Dividing by the sum makes the weights add up to 1, so the result is a
    # weighted average and the overall brightness stays the same
    kernel = kernel / kernel.sum()
    smoothed = ndimage.convolve(image, kernel, mode="reflect")

    # Store the pixel size again, so Fiji and napari show the right scale
    tifffile.imwrite(out_dir / filename, smoothed.astype(np.float32), imagej=True,
                     resolution=(1 / pixel_size, 1 / pixel_size),
                     metadata={"unit": unit})
    print(f"Saved {out_dir / filename}: center weight {kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]:.3f}, "
          f"other weights {kernel[0, 0]:.3f}, mean {smoothed.mean():.1f}")
