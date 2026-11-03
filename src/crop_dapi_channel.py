#!/usr/bin/env python3
"""
Crop a multi-channel OME-TIFF image and extract the DAPI channel
"""

import sys
import numpy as np
from skimage import io
from tifffile import imwrite


def main():
    # Parse command-line arguments
    if len(sys.argv) < 5:
        print("Usage: python crop_dapi_channel.py <x> <y> <width> <height> [channel_index] [output_file]")
        print("Example: python crop_dapi_channel.py 10 20 100 150 0 output.tif")
        print("\nDefault: channel_index=0 (DAPI), output=cropped_dapi.tif")
        sys.exit(1)

    try:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
        width = int(sys.argv[3])
        height = int(sys.argv[4])
        channel_idx = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        output_file = sys.argv[6] if len(sys.argv) > 6 else "cropped_dapi.tif"
    except ValueError:
        print("Error: x, y, width, height, and channel_index must be integers")
        sys.exit(1)

    # Load the image
    input_path = "testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"

    try:
        image = io.imread(input_path)
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Number of dimensions: {len(image.shape)}")
    print("Axis order: (Height, Width, Channels)")
    print("=" * 60)

    if len(image.shape) != 3:
        print(f"Error: Expected 3D image (Height, Width, Channels), got {len(image.shape)}D")
        sys.exit(1)

    # Image is (Height, Width, Channels)
    max_y = image.shape[0]
    max_x = image.shape[1]
    num_channels = image.shape[2]

    # Check channel validity
    if channel_idx >= num_channels:
        print(f"Error: Channel {channel_idx} not available (image has {num_channels} channels)")
        sys.exit(1)
    x_end = x + width
    y_end = y + height

    print(f"\nBounds checking:")
    print(f"  Image dimensions: {max_y} rows × {max_x} columns")
    print(f"  Number of channels: {num_channels}")
    print(f"  Requested crop: x={x} to {x_end}, y={y} to {y_end}")
    print(f"  Width: {width}, Height: {height}")

    if x < 0 or y < 0:
        print(f"Error: Negative coordinates not allowed")
        sys.exit(1)

    if x_end > max_x:
        print(f"Warning: x_end ({x_end}) exceeds image width ({max_x}). Clipping.")
        x_end = max_x

    if y_end > max_y:
        print(f"Warning: y_end ({y_end}) exceeds image height ({max_y}). Clipping.")
        y_end = max_y

    # Crop the image: image[y:y_end, x:x_end, channel] (axis order is Height, Width, Channel)
    cropped = image[y:y_end, x:x_end, channel_idx]

    print(f"\nCropping applied:")
    print(f"  Channel: {channel_idx} (DAPI)")
    print(f"  X: {x} to {x_end}, Y: {y} to {y_end}")
    print(f"  Result shape: {cropped.shape}")

    if cropped.size == 0:
        print("Error: Cropped image is empty!")
        sys.exit(1)

    # Save the result
    imwrite(output_file, cropped)
    print(f"\nSaved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
