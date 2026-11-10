"""Threshold the DAPI channel with Otsu, fill holes, number each object, and save
the label image with a LUT (color table) that gives every number its own color.

Run from the top folder of the repository with:

    uv run src/label_nuclei_lut.py

The result is saved as output/dapi_labels_lut.tif.  Open it in Fiji (File > Open):
the colors are stored in the file.  The pixel values are the object numbers, so
you can also hover over an object to read its number.
"""

import colorsys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu

input_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
out_path = Path("output/dapi_labels_lut.tif")
DAPI_INDEX = 2  # the channels are stored in the order Cy3, FITC, DAPI

with tifffile.TiffFile(input_path) as tif:
    dapi = tif.asarray()[DAPI_INDEX]
    numerator, denominator = tif.pages[0].tags["XResolution"].value
pixel_size = 1e4 * denominator / numerator  # micron per pixel

threshold = threshold_otsu(dapi)
mask = dapi > threshold
mask_filled = ndimage.binary_fill_holes(mask)  # background areas completely inside an object

# Number every group of touching pixels: 1, 2, 3, ... (0 stays background)
labels, n_objects = ndimage.label(mask_filled, structure=np.ones((3, 3)))
print(f"Otsu threshold: {threshold:.0f}")
print(f"Holes filled: {mask_filled.sum() - mask.sum()} pixels added")
print(f"Objects: {ndimage.label(mask, structure=np.ones((3, 3)))[1]} before filling, "
      f"{n_objects} after")

# A label image with a LUT can hold at most 255 objects (8 bits)
if n_objects > 255:
    raise SystemExit(f"{n_objects} objects do not fit in an 8-bit image with a LUT")

# The LUT: entry i is the color of label i.  Neighboring numbers get very
# different hues (steps of the golden ratio around the color wheel), so
# touching objects rarely look alike.  Label 0, the background, is black.
lut = np.zeros((3, 256), dtype=np.uint8)
for i in range(1, 256):
    hue = (i * 0.61803398875) % 1.0
    value = 1.0 if i % 2 else 0.8  # alternate bright and slightly darker
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, value)
    lut[:, i] = np.round(np.array([red, green, blue]) * 255)

out_path.parent.mkdir(exist_ok=True)
# Store the pixel size again, so Fiji shows the right scale
tifffile.imwrite(out_path, labels.astype(np.uint8), imagej=True,
                 resolution=(1 / pixel_size, 1 / pixel_size),
                 metadata={"unit": "micron", "LUTs": [lut], "min": 0, "max": 255})
print(f"Saved {out_path}")
