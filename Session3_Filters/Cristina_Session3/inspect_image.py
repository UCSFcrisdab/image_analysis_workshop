"""Print basic information about a TIFF image and show it in napari.

Usage: uv run Session3_Filters/Cristina_Session3/inspect_image.py <path to .tif file>
"""

import sys
import numpy as np
import tifffile
import napari

if len(sys.argv) < 2:
    print("Usage: uv run Session3_Filters/Cristina_Session3/inspect_image.py <path to .tif file>")
    sys.exit(1)

path = sys.argv[1]

with tifffile.TiffFile(path) as tif:
    image = tif.asarray()
    resolution = tif.pages[0].tags.get("XResolution")
    unit = (tif.imagej_metadata or {}).get("unit", "pixel")

# TIFF stores resolution as pixels per unit, so the pixel size is its inverse
if resolution:
    numerator, denominator = resolution.value
    pixel_size = denominator / numerator
else:
    pixel_size, unit = 1.0, "pixel"

print("=" * 60)
print(f"File:        {path}")
print(f"Shape:       {image.shape}")
print(f"Number type: {image.dtype}")
print(f"Pixel size:  {pixel_size:.4f} {unit}")
print(f"Min: {image.min()}   Max: {image.max()}")
print(f"Mean: {image.mean():.1f}   Standard deviation: {image.std():.1f}")
print("=" * 60)

viewer = napari.Viewer()
viewer.add_image(image, name=path.split("/")[-1], scale=(pixel_size, pixel_size))
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um" if unit == "micron" else None
napari.run()
