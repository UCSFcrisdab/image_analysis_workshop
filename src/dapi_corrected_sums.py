"""Background-corrected DAPI sum for every nucleus, and a histogram of the sums.

Run from the top folder of the repository, after src/watershed_nuclei_noborder.py:

    uv run src/dapi_corrected_sums.py

Steps:
    1. Keep the nuclei of the watershed labels (border nuclei are already left out)
       that are larger than 500 pixels.
    2. Grow every labeled object by 2 pixels.
    3. Background = the median DAPI value outside ALL the grown objects.
    4. For each nucleus: corrected sum = sum of the DAPI values inside its grown
       mask - background x (number of pixels in its grown mask).
    5. Plot a histogram of the corrected sums, and count saturated pixels.

Saved in output/: dapi_corrected_sums.png (histogram) and dapi_corrected_sums.csv.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.segmentation import expand_labels

image_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
labels_path = Path("output/dapi_watershed_labels_noborder.tif")
out_png = Path("output/dapi_corrected_sums.png")
out_csv = Path("output/dapi_corrected_sums.csv")
DAPI_INDEX = 2  # the channels are stored in the order Cy3, FITC, DAPI
MIN_AREA = 500  # nuclei must be larger than this (pixels)
GROW = 2  # grow every object by this many pixels

if not labels_path.exists():
    raise SystemExit(f"{labels_path} not found. Run src/watershed_nuclei_noborder.py first.")

with tifffile.TiffFile(image_path) as tif:
    dapi = tif.asarray()[DAPI_INDEX]  # the original pixel values
labels = tifffile.imread(labels_path)

# expand_labels grows every object outward by 2 pixels, but never into a
# neighboring object, so the grown masks of two touching nuclei do not overlap
# (otherwise their shared pixels would be counted twice).
grown = expand_labels(labels, distance=GROW)

# Background: the median DAPI value of all pixels that are outside every grown
# object.  The median is not pulled up by the occasional bright speck.
# We use all objects here, also the small ones that are not measured below.
background = float(np.median(dapi[grown == 0]))
print(f"Background (median outside all objects grown by {GROW} pixels): {background:.1f}")
print(f"  from {(grown == 0).sum()} pixels, {100 * (grown == 0).mean():.1f}% of the image")

# Per nucleus.  Add up the pixel values with 64-bit numbers, so nothing overflows.
area = np.bincount(labels.ravel())  # pixels per label, before growing
rows = []
for label in np.flatnonzero(area > MIN_AREA):
    if label == 0:
        continue  # 0 is the background
    region = grown == label
    n_pixels = int(region.sum())
    total = float(dapi[region].sum(dtype=np.float64))
    # The grown mask also covers some background, so subtract the background
    # for every pixel of the grown mask
    rows.append({"label": int(label), "area_px": int(area[label]),
                 "grown_area_px": n_pixels, "sum": total,
                 "corrected_sum": total - background * n_pixels})

n_objects = len(area) - 1 - int((area[1:] == 0).sum())
print(f"Objects in the labeled image: {n_objects}; larger than {MIN_AREA} pixels: {len(rows)}")
corrected = np.array([r["corrected_sum"] for r in rows])
print(f"Corrected sum: median {np.median(corrected):.4g}, range "
      f"{corrected.min():.4g} - {corrected.max():.4g}")
raw = np.array([r["sum"] for r in rows])
print(f"The correction removed on average {100 * np.mean(1 - corrected / raw):.0f}% of the raw sums")

# --- Saturation ----------------------------------------------------------------

# The camera is 16 bit, so a saturated pixel has the highest possible value, 65535.
# A saturated image also has a "plateau": many pixels with exactly the maximum
# value of the image.  We count both.
top = np.iinfo(dapi.dtype).max
at_top = int((dapi == top).sum())
at_max = int((dapi == dapi.max()).sum())
in_nuclei = int(((grown > 0) & (dapi == dapi.max())).sum())
print(f"Saturated pixels (value {top}): {at_top} of {dapi.size}")
print(f"Brightest value in the image: {dapi.max()}, reached by {at_max} pixel(s), "
      f"{in_nuclei} of them inside a nucleus")

# --- Save ----------------------------------------------------------------------

out_csv.parent.mkdir(exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    for r in rows:
        writer.writerow({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.hist(corrected, bins=20, color="#2a78d6", edgecolor="white")
ax.set_xlabel("background-corrected DAPI sum per nucleus")
ax.set_ylabel("number of nuclei")
ax.set_title(f"{len(rows)} nuclei larger than {MIN_AREA} pixels\n"
             f"background {background:.0f} per pixel, mask grown by {GROW} pixels")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(out_png, dpi=100)
print(f"Saved {out_png} and {out_csv}")
plt.show()
