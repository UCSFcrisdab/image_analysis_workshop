"""Sharpen the actin image by convolving it with a sharpening kernel.

Run from the top folder of the repository with:

    uv run src/sharpen.py

The result is saved as Session3_Filters/Cristina_Session3/output/actin_sharpen.tif.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

input_path = "testImages/Actin_cropped.tif"
out_path = Path("Session3_Filters/Cristina_Session3/output/actin_sharpen.tif")

# 12 x the center pixel minus its 8 neighbors.  The weights add up to 4, so
# flat areas come out 4 times brighter, and edges are exaggerated on top of that.
kernel = np.array([[-1, -1, -1],
                   [-1, 12, -1],
                   [-1, -1, -1]], dtype=np.float32)

# Read the image as decimal numbers (float) and its pixel size (pixels per micron)
with tifffile.TiffFile(input_path) as tif:
    image = tif.asarray().astype(np.float32)
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")
pixel_size = denominator / numerator

# Float keeps negative results; a uint16 image would wrap them around to huge values
sharpened = ndimage.convolve(image, kernel, mode="reflect")

print(f"Kernel weights add up to {kernel.sum():.0f}")
print(f"Input:  min {image.min():.1f}, max {image.max():.1f}")
print(f"Output: min {sharpened.min():.1f}, max {sharpened.max():.1f}")

# Store the pixel size again, so Fiji and napari show the right scale
out_path.parent.mkdir(parents=True, exist_ok=True)
tifffile.imwrite(out_path, sharpened, imagej=True,
                 resolution=(1 / pixel_size, 1 / pixel_size),
                 metadata={"unit": unit})
print(f"Saved {out_path}")
