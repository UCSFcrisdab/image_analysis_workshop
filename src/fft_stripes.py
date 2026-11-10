"""Make two images with vertical stripes and show them with their Fourier amplitudes.

Run from the top folder of the repository with:

    uv run src/fft_stripes.py
"""

import napari
import numpy as np

SIZE = 64
STRIPES = [30, 40]  # number of bright stripes across the image

rows, cols = np.mgrid[0:SIZE, 0:SIZE]  # cols[i, j] = j: the column of each pixel

images = []
fouriers = []
for n in STRIPES:
    # A cosine that goes through n full cycles from the left to the right edge.
    # It only depends on the column, so every row is the same: vertical stripes.
    image = np.cos(2 * np.pi * n * cols / SIZE)
    amplitude = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    images.append(image)
    fouriers.append(amplitude)

    # The strongest spots of the Fourier image (frequency 0 is at row 32, column 32)
    spots = np.argwhere(amplitude > 0.5 * amplitude.max())
    print(f"{n} stripes: bright spots at (row, column) "
          + ", ".join(f"({r}, {c})" for r, c in spots)
          + f"; amplitude {amplitude.max():.0f}")
    print(f"           distance from the center: "
          + ", ".join(f"{c - SIZE // 2:+d}" for r, c in spots) + " columns")

print()
print(f"The highest wave an image of {SIZE} pixels can hold is {SIZE // 2} stripes "
      "(one bright and one dark pixel per stripe).")

viewer = napari.Viewer()
for n, image in zip(STRIPES, images):
    viewer.add_image(image, name=f"{n} stripes")
for n, amplitude in zip(STRIPES, fouriers):
    viewer.add_image(amplitude, name=f"Fourier amplitude, {n} stripes")
viewer.grid.enabled = True
viewer.grid.shape = (2, 2)  # images on the top row, Fourier amplitudes below
napari.run()
