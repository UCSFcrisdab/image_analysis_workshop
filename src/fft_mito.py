"""Show the mitochondria image and its Fourier image side by side in napari.

Run from the top folder of the repository with:

    uv run src/fft_mito.py
"""

import napari
import numpy as np
import tifffile

image = tifffile.imread("testImages/Mito_Cropped.tif").astype(np.float32)

# The Fourier transform gives one complex number per pixel (amplitude and phase).
# fftshift moves the wave with frequency 0 (the average brightness) from the
# corner to the center of the Fourier image.
fourier = np.fft.fftshift(np.fft.fft2(image))
amplitude = np.abs(fourier)

# Amplitudes range from huge (center) to tiny (edges), so show them on a log scale.
# The 1 + avoids log(0).
log_amplitude = np.log(1 + amplitude)

print(f"Image shape: {image.shape}")
print(f"Amplitude: min {amplitude.min():.4g}, max {amplitude.max():.4g} "
      f"(at the center, row {image.shape[0] // 2}, column {image.shape[1] // 2})")
print(f"log(1 + amplitude): min {log_amplitude.min():.2f}, max {log_amplitude.max():.2f}")

viewer = napari.Viewer()
viewer.add_image(image, name="Mito_Cropped")
viewer.add_image(log_amplitude, name="log(1 + amplitude)")
viewer.grid.enabled = True  # show the layers next to each other
napari.run()
