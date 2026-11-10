"""Separate touching nuclei in the DAPI channel with the watershed algorithm.

Run from the top folder of the repository with:

    uv run src/watershed_nuclei.py

The labeled image is saved as output/dapi_watershed_labels.tif (each nucleus has
its own number) and shown in napari on top of the DAPI image.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

input_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
out_path = Path("output/dapi_watershed_labels.tif")
DAPI_INDEX = 2  # the channels are stored in the order Cy3, FITC, DAPI
SMOOTHING = 2  # blur of the distance map (pixels), so one nucleus gets one peak
MIN_DISTANCE = 10  # two nucleus centers must be at least this far apart (pixels)

with tifffile.TiffFile(input_path) as tif:
    dapi = tif.asarray()[DAPI_INDEX]
    numerator, denominator = tif.pages[0].tags["XResolution"].value
pixel_size = 1e4 * denominator / numerator  # micron per pixel

threshold = threshold_otsu(dapi)
mask = ndimage.binary_fill_holes(dapi > threshold)
touching = np.ones((3, 3))  # pixels that touch at a corner belong together
labels_before, n_before = ndimage.label(mask, structure=touching)

# The distance map: for every nucleus pixel, how far it is from the background.
# It is highest in the middle of a round nucleus, so each nucleus gets a peak,
# and two touching nuclei get two peaks with a "saddle" where they meet.
distance = ndimage.distance_transform_edt(mask)
distance = ndimage.gaussian_filter(distance, SMOOTHING)

# The peaks are the starting points ("markers") of the flooding
peaks = peak_local_max(distance, min_distance=MIN_DISTANCE, labels=labels_before,
                       exclude_border=False)
markers = np.zeros(mask.shape, dtype=np.int32)
markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
# Objects too small to have a peak (like single-pixel specks) still get a label
unmarked = (labels_before > 0) & (ndimage.maximum(markers > 0, labels_before,
            index=np.arange(n_before + 1))[labels_before] == 0)
extra, n_extra = ndimage.label(unmarked, structure=touching)
markers[extra > 0] = extra[extra > 0] + len(peaks)

# Watershed: imagine -distance as a landscape with a valley at every marker.
# Water rises from all the markers at the same time; where the water from two
# valleys meets, a boundary is drawn.  mask=mask keeps it inside the nuclei.
labels = watershed(-distance, markers, mask=mask, connectivity=2)
n_after = labels.max()

sizes = np.bincount(labels.ravel())[1:]
sizes_before = np.bincount(labels_before.ravel())[1:]
print(f"Otsu threshold {threshold:.0f}, holes filled")
print(f"Objects before watershed: {n_before}")
print(f"Objects after watershed:  {n_after}")
split = [(i + 1) for i in range(n_before)
         if len(np.unique(labels[labels_before == i + 1])) > 1]
print(f"Objects that were split: {len(split)}")
for i in split:
    parts = np.unique(labels[labels_before == i])
    centroid = ndimage.center_of_mass(labels_before == i)
    print(f"  object {i} at row {centroid[0]:.0f}, column {centroid[1]:.0f}: "
          f"{sizes_before[i - 1]} pixels -> {len(parts)} parts of "
          f"{', '.join(str(sizes[p - 1]) for p in parts)} pixels")
print(f"Size after: smallest {sizes.min()}, median {int(np.median(sizes))}, largest {sizes.max()}")

out_path.parent.mkdir(exist_ok=True)
# Store the pixel size again, so Fiji shows the right scale
tifffile.imwrite(out_path, labels.astype(np.uint16), imagej=True,
                 resolution=(1 / pixel_size, 1 / pixel_size),
                 metadata={"unit": "micron"})
print(f"Saved {out_path}")

scale = (pixel_size, pixel_size)
viewer = napari.Viewer()
viewer.add_image(dapi, name="DAPI", scale=scale)
viewer.add_labels(labels, name="watershed labels", scale=scale)
napari.run()
