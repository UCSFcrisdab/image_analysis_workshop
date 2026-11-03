#!/usr/bin/env python3
"""
Examine nucleus image with flexible filename input
"""

import sys
import numpy as np
from skimage import io
import napari


def main():
    # Check if filename was provided
    if len(sys.argv) < 2:
        print("Usage: python examine_nucleus_image_flexible.py <filename>")
        print("Example: python examine_nucleus_image_flexible.py testImages/nucleus_cropped_2.tif")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Load the image
        image = io.imread(image_path)
    except FileNotFoundError:
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    # Print basic image information
    print("=" * 50)
    print(f"File: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print("=" * 50)

    # Print first 10 rows and 10 columns
    print("\nFirst 10 rows and 10 columns as a matrix:")
    print(image[:10, :10])
    print()

    # Display in napari
    viewer = napari.Viewer()
    viewer.add_image(image, name=image_path)
    napari.run()


if __name__ == "__main__":
    main()
