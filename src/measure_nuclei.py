"""Measure every nucleus: size, roundness, area in square microns, and DAPI brightness.

Run from the top folder of the repository, after src/watershed_nuclei_noborder.py:

    uv run src/measure_nuclei.py

It reads the labeled image output/dapi_watershed_labels_noborder.tif (every pixel
holds the number of its nucleus) and the ORIGINAL DAPI pixel values, and saves
one row per nucleus in output/nuclei_measurements.csv.
"""

import csv
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.measure import regionprops

image_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
labels_path = Path("output/dapi_watershed_labels_noborder.tif")
out_path = Path("output/nuclei_measurements.csv")
DAPI_INDEX = 2  # the channels are stored in the order Cy3, FITC, DAPI
MIN_SIZE = 100  # objects smaller than this (in pixels) are not nuclei: specks and debris

if not labels_path.exists():
    raise SystemExit(f"{labels_path} not found. Run src/watershed_nuclei_noborder.py first.")

with tifffile.TiffFile(image_path) as tif:
    dapi = tif.asarray()[DAPI_INDEX]  # the original pixel values, not a copy that was changed
    numerator, denominator = tif.pages[0].tags["XResolution"].value
pixel_size = 1e4 * denominator / numerator  # micron per pixel
labels = tifffile.imread(labels_path)

rows = []
for region in regionprops(labels, intensity_image=dapi):
    if region.area < MIN_SIZE:
        continue  # skip specks
    mask = labels == region.label
    rows.append({
        "label": region.label,
        "area_pixels": int(region.area),
        # One pixel covers pixel_size x pixel_size micron, so multiply the pixel count by that
        "area_um2": region.area * pixel_size**2,
        # The nucleus is fitted with an ellipse; roundness = long axis / short axis.
        # 1.0 is a perfect circle, and the larger the number, the more elongated.
        "roundness": region.axis_major_length / region.axis_minor_length,
        # The brightness of the ORIGINAL DAPI pixels under the mask of this nucleus
        "sum": float(dapi[mask].sum(dtype=np.float64)),
        "mean": float(region.intensity_mean),
        "std": float(region.intensity_std),  # spread of the pixel values (n, not n - 1)
    })

n_all = len(np.unique(labels)) - 1
print(f"Objects in the label image: {n_all}")
print(f"Nuclei of at least {MIN_SIZE} pixels ({MIN_SIZE * pixel_size**2:.1f} square micron): "
      f"{len(rows)}   (left out: {n_all - len(rows)} smaller objects)")

for key, unit in [("area_um2", "square micron"), ("roundness", "long / short axis"),
                  ("mean", "mean DAPI value"), ("sum", "sum of DAPI values")]:
    values = np.array([r[key] for r in rows])
    print(f"{key:>10}: median {np.median(values):.4g}, "
          f"range {values.min():.4g} - {values.max():.4g}   ({unit})")

out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    for r in rows:
        writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print(f"Saved {out_path}")
