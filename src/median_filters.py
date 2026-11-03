"""Apply median filters with three different footprints to a noisy nucleus image.

Run from the top folder of the repository with:

    uv run src/median_filters.py

Results are saved in output/ as median_plus3.tif, median_3x3.tif and median_5x5.tif.
"""

from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import median

input_path = "testImages/Nucleus_cropped_noise.tif"
out_dir = Path("output")

# The footprint marks which neighbors take part (1 = used, 0 = ignored)
footprints = {
    "median_plus3.tif": np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=bool),
    "median_3x3.tif": np.ones((3, 3), dtype=bool),
    "median_5x5.tif": np.ones((5, 5), dtype=bool),
}

# Read the image and its pixel size (TIFF stores pixels per micron)
with tifffile.TiffFile(input_path) as tif:
    image = tif.asarray()
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")
pixel_size = denominator / numerator

print(f"Input:  {input_path}, shape {image.shape}, type {image.dtype}, "
      f"pixel size {pixel_size:.4f} {unit}")
print(f"        standard deviation {image.std():.1f}")

out_dir.mkdir(exist_ok=True)
for filename, footprint in footprints.items():
    # Each output pixel becomes the middle value of the pixels under the footprint
    filtered = median(image, footprint=footprint)

    # Store the pixel size again, so Fiji and napari show the right scale
    tifffile.imwrite(out_dir / filename, filtered, imagej=True,
                     resolution=(1 / pixel_size, 1 / pixel_size),
                     metadata={"unit": unit})
    print(f"Saved {out_dir / filename}: {footprint.sum()} pixels per median, "
          f"standard deviation {filtered.std():.1f}")
