"""Segment the nuclei in the DAPI channel with a fixed threshold.

Run from the top folder of the repository with:

    uv run src/segment_dapi.py

The original image is only read, never changed.  Results are saved in output/:
dapi_mask.tif (255 = nucleus, 0 = background) and dapi_labels.tif (each nucleus
has its own number: 1, 2, 3, ...).
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage

input_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
out_dir = Path("output")

THRESHOLD = 1900
DAPI_INDEX = 2  # the channels are stored in the order Cy3, FITC, DAPI

with tifffile.TiffFile(input_path) as tif:
    stack = tif.asarray()  # shape (channels, rows, columns)
    # Pixel size: the file stores pixels per centimeter
    numerator, denominator = tif.pages[0].tags["XResolution"].value
pixel_size = 1e4 * denominator / numerator  # micron per pixel

dapi = stack[DAPI_INDEX]
print(f"Image stack: {stack.shape}, type {stack.dtype}, pixel size {pixel_size:.4f} micron")
print(f"DAPI channel: min {dapi.min()}, max {dapi.max()}")

# Pixels brighter than the threshold are nucleus
mask = dapi > THRESHOLD

# Give every group of touching nucleus pixels its own number.  Pixels that only
# touch at a corner count as touching (8-connected).
labels, n_nuclei = ndimage.label(mask, structure=np.ones((3, 3)))
areas = np.bincount(labels.ravel())[1:]  # pixels per nucleus
print(f"Threshold {THRESHOLD}: {mask.sum()} pixels ({100 * mask.mean():.1f}%) "
      f"in {n_nuclei} objects")
print(f"Object size in pixels: smallest {areas.min()}, median {int(np.median(areas))}, "
      f"largest {areas.max()}; objects of 10 pixels or less: {(areas <= 10).sum()}")

out_dir.mkdir(exist_ok=True)
# Store the pixel size again, so Fiji and napari show the right scale
scale = dict(imagej=True, resolution=(1 / pixel_size, 1 / pixel_size),
             metadata={"unit": "micron"})
tifffile.imwrite(out_dir / "dapi_mask.tif", mask.astype(np.uint8) * 255, **scale)
tifffile.imwrite(out_dir / "dapi_labels.tif", labels.astype(np.uint16), **scale)
print(f"Saved {out_dir / 'dapi_mask.tif'} and {out_dir / 'dapi_labels.tif'}")

viewer = napari.Viewer()
viewer.add_image(dapi, name="DAPI", scale=(pixel_size, pixel_size))
viewer.add_labels(labels, name=f"nuclei (threshold {THRESHOLD})",
                  scale=(pixel_size, pixel_size))
napari.run()
