"""Find where a small template matches the mitochondria image, with plain and with
normalized cross-correlation, and show the matches in napari.

Run from the top folder of the repository with:

    uv run src/xcorr.py

Results are saved in Session3_Filters/Cristina_Session3/output/:
xcorr_plain.tif, xcorr_norm.tif and peaks.csv.
"""

import csv
from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy import signal
from skimage.feature import match_template, peak_local_max

image_path = "testImages/Mito_Cropped.tif"
template_path = "testImages/Mito_template.tif"
out_dir = Path("Session3_Filters/Cristina_Session3/output")

# Read both images as decimal numbers (float), and the pixel size of the image
with tifffile.TiffFile(image_path) as tif:
    image = tif.asarray().astype(np.float32)
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")
pixel_size = denominator / numerator
template = tifffile.imread(template_path).astype(np.float32)

print(f"Image:    {image_path}, shape {image.shape}")
print(f"Template: {template_path}, shape {template.shape}")

# Slide the template over the image; at each position multiply the overlapping
# pixels and add them up.  mode="same" keeps the output the size of the image.
xcorr = signal.correlate(image, template, mode="same")

# Position of the highest value: where the template "matches" best
row, col = np.unravel_index(np.argmax(xcorr), xcorr.shape)
print(f"Cross-correlation: min {xcorr.min():.4g}, max {xcorr.max():.4g}")
print(f"Maximum at row {row}, column {col}")

out_dir.mkdir(parents=True, exist_ok=True)


def save(filename, data):
    # Store the pixel size again, so Fiji and napari show the right scale
    tifffile.imwrite(out_dir / filename, data.astype(np.float32), imagej=True,
                     resolution=(1 / pixel_size, 1 / pixel_size),
                     metadata={"unit": unit})
    print(f"Saved {out_dir / filename}")


save("xcorr_plain.tif", xcorr)

# Normalized cross-correlation: at each position the template and the image patch
# are both corrected for their own brightness and contrast, so only the shape is
# compared.  Scores go from -1 to 1 (1 = perfect match).  pad_input=True keeps
# the output the size of the image, with each score at the template's center.
xcorr_norm = match_template(image, template, pad_input=True)
print(f"\nNormalized cross-correlation: min {xcorr_norm.min():.2f}, max {xcorr_norm.max():.2f}")
save("xcorr_norm.tif", xcorr_norm)

# Peaks: local maxima scoring at least 0.8, and at least 4 pixels apart
peaks = peak_local_max(xcorr_norm, threshold_abs=0.8, min_distance=4)
print(f"Found {len(peaks)} peaks with a score of at least 0.8")

with open(out_dir / "peaks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["row", "column", "score"])
    for r, c in peaks:
        writer.writerow([r, c, round(float(xcorr_norm[r, c]), 3)])
print(f"Saved {out_dir / 'peaks.csv'}")

# Show the image with a red circle on every peak.  The points get the same
# scale as the image, so the circles land on the right pixels.
scale = (pixel_size, pixel_size)
viewer = napari.Viewer()
viewer.add_image(image, name="Mito_Cropped", scale=scale)
viewer.add_points(peaks, name="template matches", scale=scale, size=11,
                  face_color="transparent", border_color="red")
napari.run()
