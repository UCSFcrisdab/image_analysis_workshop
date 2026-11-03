"""Save an image as JPEG once and 100 times, and show what JPEG compression changed."""

import numpy as np
import tifffile
import imageio.v3 as iio
import napari

input_path = "testImages/Cell_cropped_2.tif"
jpg_path = "Cell_cropped_2.jpg"
n_saves = 100

# Read the original and its pixel size (stored as pixels per micron in the TIFF)
with tifffile.TiffFile(input_path) as tif:
    original = tif.asarray()
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    pixel_size = denominator / numerator

print(f"Original: shape {original.shape}, type {original.dtype}, pixel size {pixel_size:.4f} micron")

# Save as JPEG (quality goes from 1 = worst to 100 = best) and read it back
iio.imwrite(jpg_path, original, quality=75)
jpg_once = iio.imread(jpg_path)

# Re-open and re-save the same JPEG file, as if editing and saving it many times
jpg = jpg_once
for _ in range(n_saves - 1):
    iio.imwrite(jpg_path, jpg, quality=75)
    jpg = iio.imread(jpg_path)


def difference_from_original(image):
    # Convert to signed integers first, otherwise negative differences wrap around to large values
    diff = image.astype(np.int16) - original.astype(np.int16)
    print(f"  min {diff.min()}, max {diff.max()}, mean absolute {np.abs(diff).mean():.2f}")
    return diff


print("Difference after 1 save:")
diff_once = difference_from_original(jpg_once)
print(f"Difference after {n_saves} saves:")
diff_many = difference_from_original(jpg)

# JPEG files cannot store the pixel size, so give it to napari for every layer
scale = (pixel_size, pixel_size)
viewer = napari.Viewer()
viewer.add_image(original, name="original", rgb=True, scale=scale)
viewer.add_image(jpg_once, name="jpg (1 save)", rgb=True, scale=scale)
viewer.add_image(jpg, name=f"jpg ({n_saves} saves)", rgb=True, scale=scale)
for diff, label in [(diff_once, "1 save"), (diff_many, f"{n_saves} saves")]:
    viewer.add_image(diff, name=[f"difference R ({label})", f"difference G ({label})",
                                 f"difference B ({label})"],
                     channel_axis=2, colormap=["red", "green", "blue"],
                     contrast_limits=(-60, 60), scale=scale)
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"
napari.run()
