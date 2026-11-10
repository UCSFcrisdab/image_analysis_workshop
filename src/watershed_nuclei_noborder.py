"""Separate touching nuclei in the DAPI channel with the watershed algorithm, and
leave out the nuclei that touch the border of the image.

Run from the top folder of the repository with:

    uv run src/watershed_nuclei_noborder.py

The labeled image is saved as output/dapi_watershed_labels_noborder.tif (each
nucleus has its own number) and shown in napari on top of the DAPI image.

The steps:
    1. Make a black-and-white mask of the nuclei (Otsu threshold, fill holes).
    2. Find the middle of every nucleus (with a distance map).
    3. Run the watershed from those middles, so touching nuclei get separated.
    4. Remove the nuclei that touch the edge of the image.
    5. Save the result and show it in napari.
"""

# --- The tools we need ---------------------------------------------------------

from pathlib import Path  # file and folder names

import napari  # the image viewer
import numpy as np  # arrays of numbers (an image is an array)
import tifffile  # reads and writes TIFF files
from scipy import ndimage  # filters, labeling, distance maps
from skimage.feature import peak_local_max  # finds the highest spots in an image
from skimage.filters import threshold_otsu  # automatic threshold
from skimage.segmentation import clear_border, relabel_sequential, watershed

# --- Settings you can change ---------------------------------------------------

input_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
out_path = Path("output/dapi_watershed_labels_noborder.tif")
DAPI_INDEX = 2  # the channels are stored in the order Cy3, FITC, DAPI
# Two settings that decide how readily touching nuclei are split:
SMOOTHING = 2  # blur of the distance map (pixels), so one nucleus gets one peak
MIN_DISTANCE = 10  # two nucleus centers must be at least this far apart (pixels)

# --- 1. Read the image ---------------------------------------------------------

with tifffile.TiffFile(input_path) as tif:
    # The file holds 3 images (one per channel); keep only the DAPI one.
    # This only reads the file, the original is never changed.
    dapi = tif.asarray()[DAPI_INDEX]
    # The size of a pixel is stored as "pixels per centimeter"
    numerator, denominator = tif.pages[0].tags["XResolution"].value
pixel_size = 1e4 * denominator / numerator  # micron per pixel

# --- 2. Make the mask: nucleus (True) or background (False) --------------------

# Otsu picks the threshold that best separates dark pixels from bright pixels.
# Every pixel brighter than the threshold is nucleus.  Then fill_holes fills any
# dark area that is completely inside a nucleus, so that nuclei are solid.
threshold = threshold_otsu(dapi)
mask = ndimage.binary_fill_holes(dapi > threshold)

# Give every separate object in the mask its own number (1, 2, 3, ...).  Pixels
# that touch only at a corner count as touching, so they belong to one object.
touching = np.ones((3, 3))
labels_before, n_before = ndimage.label(mask, structure=touching)

# --- 3. Find the middle of every nucleus ---------------------------------------

# The distance map: for every nucleus pixel, how far it is from the background.
# It is highest in the middle of a round nucleus, so each nucleus gets a peak,
# and two touching nuclei get two peaks with a "saddle" where they meet.
distance = ndimage.distance_transform_edt(mask)
# A little blur removes small bumps, so that one nucleus does not get two peaks
distance = ndimage.gaussian_filter(distance, SMOOTHING)

# The peaks are the starting points ("markers") of the flooding below.
# Each marker gets its own number, 1, 2, 3, ...  min_distance stops two peaks
# from being closer together than a typical nucleus is wide.
peaks = peak_local_max(distance, min_distance=MIN_DISTANCE, labels=labels_before,
                       exclude_border=False)
markers = np.zeros(mask.shape, dtype=np.int32)
markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)

# Some objects are too small to have a peak (like single-pixel specks).  Give
# each of them a marker too, so that they are not lost: "unmarked" is every
# mask pixel in an object that has no marker yet.
unmarked = (labels_before > 0) & (ndimage.maximum(markers > 0, labels_before,
            index=np.arange(n_before + 1))[labels_before] == 0)
extra, n_extra = ndimage.label(unmarked, structure=touching)
markers[extra > 0] = extra[extra > 0] + len(peaks)

# --- 4. Watershed --------------------------------------------------------------

# Imagine -distance as a landscape with a valley at every marker.  Water rises
# from all the markers at the same time; where the water from two valleys meets,
# a boundary is drawn.  mask=mask keeps it inside the nuclei.  connectivity=2
# lets the flood pass between pixels that touch at a corner, matching the
# labeling above (otherwise a few pixels would be left without a number).
labels = watershed(-distance, markers, mask=mask, connectivity=2)
n_after = labels.max()

# Count the pixels of every nucleus: sizes[0] is nucleus 1, sizes[1] is nucleus 2, ...
sizes = np.bincount(labels.ravel())[1:]
sizes_before = np.bincount(labels_before.ravel())[1:]
print(f"Otsu threshold {threshold:.0f}, holes filled")
print(f"Objects before watershed: {n_before}")
print(f"Objects after watershed:  {n_after}")

# Which of the original objects did the watershed cut into more than one nucleus?
split = [(i + 1) for i in range(n_before)
         if len(np.unique(labels[labels_before == i + 1])) > 1]
print(f"Objects that were split: {len(split)}")
for i in split:
    parts = np.unique(labels[labels_before == i])
    centroid = ndimage.center_of_mass(labels_before == i)
    print(f"  object {i} at row {centroid[0]:.0f}, column {centroid[1]:.0f}: "
          f"{sizes_before[i - 1]} pixels -> {len(parts)} parts of "
          f"{', '.join(str(sizes[p - 1]) for p in parts)} pixels")

# --- 5. Remove the nuclei that touch the border --------------------------------

# Remove every nucleus that has at least one pixel in the first or last row or
# column of the image: it is cut off by the edge, so its size and shape are wrong.
# The remaining nuclei are numbered again from 1, without gaps.
# (The labels along the four edges are the nuclei that touch the border; 0 is
# the background, so it is left out.)
border_labels = np.setdiff1d(np.unique(np.concatenate(
    [labels[0], labels[-1], labels[:, 0], labels[:, -1]])), [0])
removed_sizes = sizes[border_labels - 1]
labels, _, _ = relabel_sequential(clear_border(labels))
sizes = np.bincount(labels.ravel())[1:]
print(f"Nuclei touching the border (removed): {len(border_labels)}, "
      f"sizes in pixels: {', '.join(str(s) for s in sorted(removed_sizes))}")
print(f"Nuclei left: {labels.max()}")
print(f"Size after: smallest {sizes.min()}, median {int(np.median(sizes))}, largest {sizes.max()}")
print(f"Objects of 100 pixels or less left: {(sizes <= 100).sum()}")

# --- 6. Save the result --------------------------------------------------------

out_path.parent.mkdir(exist_ok=True)
# The value of every pixel is the number of its nucleus (0 = background).
# Store the pixel size again, so Fiji shows the right scale.
tifffile.imwrite(out_path, labels.astype(np.uint16), imagej=True,
                 resolution=(1 / pixel_size, 1 / pixel_size),
                 metadata={"unit": "micron"})
print(f"Saved {out_path}")

# --- 7. Show it in napari ------------------------------------------------------

# The same scale for both layers, so that they line up and distances are in micron
scale = (pixel_size, pixel_size)
viewer = napari.Viewer()
viewer.add_image(dapi, name="DAPI", scale=scale)
viewer.add_labels(labels, name="watershed labels, no border objects", scale=scale)
napari.run()
