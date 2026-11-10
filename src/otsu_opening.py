"""Threshold the DAPI channel with Otsu, clean up with erosion then dilation, and plot object sizes.

Run from the top folder of the repository with:

    uv run src/otsu_opening.py

The histogram is shown in a window and saved as output/otsu_opening_sizes.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu

input_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
out_path = Path("output/otsu_opening_sizes.png")
DAPI_INDEX = 2  # the channels are stored in the order Cy3, FITC, DAPI
ROUNDS = 2

with tifffile.TiffFile(input_path) as tif:
    dapi = tif.asarray()[DAPI_INDEX]

threshold = threshold_otsu(dapi)
mask = dapi > threshold

# Erosion peels one layer of pixels off every object, so small specks disappear.
# Dilation then adds a layer back, so the survivors get their size back.
# Doing both, in this order, is called an "opening".  The default structuring
# element is a plus shape (a pixel and its 4 neighbors up, down, left, right).
eroded = ndimage.binary_erosion(mask, iterations=ROUNDS)
opened = ndimage.binary_dilation(eroded, iterations=ROUNDS)


def object_sizes(binary):
    labels, n = ndimage.label(binary, structure=np.ones((3, 3)))  # touching corners count
    return np.bincount(labels.ravel())[1:], n


sizes_before, n_before = object_sizes(mask)
sizes_after, n_after = object_sizes(opened)

print(f"Otsu threshold: {threshold:.0f}")
print(f"Before: {n_before} objects, {mask.sum()} pixels")
print(f"After {ROUNDS} erosions and {ROUNDS} dilations: {n_after} objects, {opened.sum()} pixels")
print(f"Object size in pixels after: smallest {sizes_after.min()}, "
      f"median {int(np.median(sizes_after))}, largest {sizes_after.max()}")
print(f"Objects removed: {n_before - n_after}; pixels removed: {mask.sum() - opened.sum()}")

fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(0, max(sizes_before.max(), sizes_after.max()), 40)
ax.hist(sizes_before, bins=bins, color="#c8c8c8", label=f"before ({n_before} objects)")
ax.hist(sizes_after, bins=bins, color="#2a78d6", alpha=0.85,
        label=f"after opening ({n_after} objects)")
ax.set_xlabel("object size (pixels)")
ax.set_ylabel("number of objects")
ax.set_title(f"DAPI, Otsu threshold {threshold:.0f}, then {ROUNDS} erosions and "
             f"{ROUNDS} dilations")
ax.legend(frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()

out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=100)
print(f"Saved {out_path}")
plt.show()
