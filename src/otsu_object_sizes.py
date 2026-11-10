"""Threshold the DAPI channel with Otsu, count the objects, and plot their sizes.

Run from the top folder of the repository with:

    uv run src/otsu_object_sizes.py

The histogram is shown in a window and saved as output/otsu_object_sizes.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu

input_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
out_path = Path("output/otsu_object_sizes.png")
DAPI_INDEX = 2  # the channels are stored in the order Cy3, FITC, DAPI

with tifffile.TiffFile(input_path) as tif:
    dapi = tif.asarray()[DAPI_INDEX]
    numerator, denominator = tif.pages[0].tags["XResolution"].value
pixel_size = 1e4 * denominator / numerator  # micron per pixel

# Otsu picks the threshold that best separates the pixels into a dark and a bright group
threshold = threshold_otsu(dapi)
mask = dapi > threshold

# Number each group of touching pixels (8-connected), then count the pixels of each
labels, n_objects = ndimage.label(mask, structure=np.ones((3, 3)))
sizes = np.bincount(labels.ravel())[1:]

print(f"Otsu threshold: {threshold:.0f}")
print(f"Number of objects: {n_objects}")
print(f"Object size in pixels: smallest {sizes.min()}, median {int(np.median(sizes))}, "
      f"largest {sizes.max()}")
print(f"Objects of 10 pixels or less: {(sizes <= 10).sum()}; "
      f"more than 500 pixels: {(sizes > 500).sum()}")

fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))
# Left: all objects.  Right: the same data with a log scale on the sizes, so that
# the many tiny objects and the few huge ones are both visible
ax_lin.hist(sizes, bins=40, color="#2a78d6")
ax_lin.set_xlabel("object size (pixels)")
ax_lin.set_title(f"All {n_objects} objects")
bins = np.logspace(0, np.log10(sizes.max()), 40)
ax_log.hist(sizes, bins=bins, color="#2a78d6")
ax_log.set_xscale("log")
ax_log.set_xlabel("object size (pixels, log scale)")
ax_log.set_title("Same objects, log scale")
for ax in (ax_lin, ax_log):
    ax.set_ylabel("number of objects")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(f"DAPI, Otsu threshold {threshold:.0f}  "
             f"(1 pixel = {pixel_size:.3f} x {pixel_size:.3f} micron)", fontsize=12)
fig.tight_layout()

out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=100)
print(f"Saved {out_path}")
plt.show()
