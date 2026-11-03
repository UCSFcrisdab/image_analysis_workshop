"""Blur the actin image with two Gaussian kernels, subtract each blur from the input,
and sharpen the input by adding that difference back (unsharp masking).

Run from the top folder of the repository with:

    uv run src/gaussian_unsharp.py

Results are saved in Session3_Filters/Cristina_Session3/output/:
gauss_<size>.tif, unsharp_<size>.tif and sharpened_<size>_<amount>.tif,
for size 5x5 and 21x21 and amount 0.5, 1 and 2.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

input_path = "testImages/Actin_cropped.tif"
out_dir = Path("Session3_Filters/Cristina_Session3/output")

# sigma sets how wide the bell curve is; radius cuts it off, so the kernel is
# (2 x radius + 1) pixels wide: radius 2 -> 5 x 5, radius 10 -> 21 x 21
gaussians = {
    "5x5": {"sigma": 1.0, "radius": 2},
    "21x21": {"sigma": 3.5, "radius": 10},
}

# How much of the detail (input minus blur) to add back to the input
amounts = [0.5, 1, 2]

# Read the image as decimal numbers (float) and its pixel size (pixels per micron)
with tifffile.TiffFile(input_path) as tif:
    image = tif.asarray().astype(np.float32)
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")
pixel_size = denominator / numerator


def save(filename, data):
    # Store the pixel size again, so Fiji and napari show the right scale
    tifffile.imwrite(out_dir / filename, data.astype(np.float32), imagej=True,
                     resolution=(1 / pixel_size, 1 / pixel_size),
                     metadata={"unit": unit})
    print(f"Saved {filename:<24} min {data.min():8.1f}   max {data.max():8.1f}")


print(f"Input: {input_path}, min {image.min():.1f}, max {image.max():.1f}, "
      f"pixel size {pixel_size:.4f} {unit}")
out_dir.mkdir(parents=True, exist_ok=True)

for size, settings in gaussians.items():
    blurred = ndimage.gaussian_filter(image, mode="reflect", **settings)
    # What is left after removing the blur: the fine detail (edges and thin fibers)
    detail = image - blurred
    save(f"gauss_{size}.tif", blurred)
    save(f"unsharp_{size}.tif", detail)

    # Unsharp masking: add the detail back on top of the original to sharpen it
    for amount in amounts:
        save(f"sharpened_{size}_{amount:g}.tif", image + amount * detail)
