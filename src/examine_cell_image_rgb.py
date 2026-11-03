#!/usr/bin/env python3
"""
Examine RGB cell image: display sampled R,G,B values in a table and visualize in napari
"""

import numpy as np
from skimage import io
import napari


def main():
    # Load the image
    image_path = "testImages/Cell_cropped_2.tif"
    image = io.imread(image_path)

    print("=" * 60)
    print(f"File: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print("=" * 60)

    # Sample approximately 3% of pixels with regular spacing
    # Stride of ~6 gives ~2.8% coverage (1/36 ≈ 0.028)
    stride = 6
    sampled_rows = image[::stride, ::stride]

    # Extract R, G, B values
    if len(image.shape) == 3 and image.shape[2] == 3:
        r_values = sampled_rows[:, :, 0]
        g_values = sampled_rows[:, :, 1]
        b_values = sampled_rows[:, :, 2]

        # Flatten and create table
        r_flat = r_values.flatten()
        g_flat = g_values.flatten()
        b_flat = b_values.flatten()

        print(f"\nR, G, B values for {len(r_flat)} sampled pixels ({stride}×{stride} stride):")
        print("-" * 40)
        print(f"{'Pixel #':<8} {'R':<8} {'G':<8} {'B':<8}")
        print("-" * 40)

        for i in range(len(r_flat)):
            print(f"{i:<8} {r_flat[i]:<8} {g_flat[i]:<8} {b_flat[i]:<8}")
        print()
    else:
        print("Error: Image does not appear to be RGB (3-channel color)")
        return

    # Display in napari
    viewer = napari.Viewer()
    viewer.add_image(image, name="Cell_cropped_2", rgb=True)
    napari.run()


if __name__ == "__main__":
    main()
