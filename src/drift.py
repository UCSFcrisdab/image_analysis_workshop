"""Shift the mitochondria image by a known amount and measure the shift back with
phase cross-correlation.

Run from the top folder of the repository with:

    uv run src/drift.py
"""

import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation

image_path = "testImages/Mito_Cropped.tif"
true_shift = (7, 4)  # (rows down, columns right)

image = tifffile.imread(image_path).astype(np.float32)

# Move every pixel 7 rows down and 4 columns right; pixels pushed off one edge
# come back in on the opposite edge
shifted = np.roll(image, shift=true_shift, axis=(0, 1))

# Returns the shift needed to move "shifted" back onto "image", so it has the
# opposite sign of the shift we applied
estimated, _, _ = phase_cross_correlation(image, shifted)

print(f"Image: {image_path}, shape {image.shape}")
print(f"Applied shift (rows, columns):   {true_shift}")
print(f"Estimated shift (rows, columns): ({estimated[0]:g}, {estimated[1]:g})"
      "   <- the shift that moves the copy back")
print(f"So the copy moved by:            ({-estimated[0]:g}, {-estimated[1]:g})")
