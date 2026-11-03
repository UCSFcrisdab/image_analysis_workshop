"""Load testim.ics, compare it with its header metadata, and show it in napari."""

from bioio import BioImage
import napari

img = BioImage("testImages/testim.ics")

print("Full bioio shape (T, C, Z, Y, X):", img.shape)

# Keep only the Z, Y, X axes (there is just 1 time point and 1 channel)
image = img.get_image_data("ZYX")
pixel_sizes = img.physical_pixel_sizes

print("=" * 60)
print("Header says:  X=175, Y=105, Z=2, 16-bit unsigned integer")
print(f"Loaded image: shape (Z, Y, X) = {image.shape}, type = {image.dtype}")
print(f"Pixel sizes (Z, Y, X): {pixel_sizes.Z}, {pixel_sizes.Y}, {pixel_sizes.X}")
print(f"Min value: {image.min()}, max value: {image.max()}")

if image.shape == (2, 105, 175) and image.dtype == "uint16":
    print("Matches the header metadata.")
else:
    print("Does NOT match the header metadata.")
print("=" * 60)

scale = [s if s else 1.0 for s in (pixel_sizes.Z, pixel_sizes.Y, pixel_sizes.X)]
viewer = napari.Viewer()
viewer.add_image(image, name="testim", scale=scale)
napari.run()
