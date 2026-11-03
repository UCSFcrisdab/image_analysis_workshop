#!/usr/bin/env python3
"""
Examine nucleus image with strided sampling: display data, type, and visualize in napari
"""

import numpy as np
from skimage import io
import napari


def main():
    # Load the image
    image_path = "testImages/nucleus_cropped_1.tif"
    image = io.imread(image_path)

    # Print basic image information
    print("=" * 50)
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print("=" * 50)

    # Print 10 rows and 10 columns, skipping 2 rows and 2 columns each time
    print("\n10 rows and 10 columns (skipping 2 each time):")
    print(image[0:30:3, 0:30:3])
    print()

    # Display in napari
    viewer = napari.Viewer()
    viewer.add_image(image, name="nucleus_cropped_1")
    napari.run()


if __name__ == "__main__":
    main()
