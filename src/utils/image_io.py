import cv2
import numpy as np
from pathlib import Path

class RawImageReader:
    """Handles reading and parsing of raw image files."""

    def __init__(self, width=1920, height=1280, bits=12):
        self.width = width
        self.height = height
        self.bits = bits
        self.max_value = (1 << bits) - 1

    def read_raw(self, file_path):
        """
        Read a raw image file.

        Args:
            file_path (str): Path to the raw image file

        Returns:
            numpy.ndarray: Raw image data as 2D array
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Raw file not found: {file_path}")

        # Read raw bytes
        raw_data = np.fromfile(file_path, dtype=np.uint16)

        # Reshape to image dimensions
        try:
            image = raw_data.reshape((self.height, self.width))
        except ValueError:
            raise ValueError(f"Raw file size doesn't match dimensions {self.width}x{self.height}")

        # Normalize to specified bit depth
        image = image & self.max_value

        return image

class ImageWriter:
    """Handles saving of processed images in various formats."""

    @staticmethod
    def save_image(image, file_path, bit_depth=8):
        """
        Save processed image to file.

        Args:
            image (numpy.ndarray): Image data
            file_path (str): Output file path
            bit_depth (int): Bit depth of output image
        """
        file_path = Path(file_path)

        # Create output directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize image to specified bit depth
        max_value = (1 << bit_depth) - 1
        normalized = (image * max_value).clip(0, max_value).astype(np.uint8)

        # Save image based on extension
        cv2.imwrite(str(file_path), normalized)

def load_image(file_path):
    """
    Load an image from a file. Supports both raw and common image formats.

    Args:
        file_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded image.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if it's a raw file (you can modify the extension check as needed)
    if file_path.suffix == '.raw':
        # Assuming raw image is 12-bit Bayer format with known dimensions
        raw_reader = RawImageReader(width=1920, height=1280, bits=12)
        return raw_reader.read_raw(str(file_path))
    else:
        # Load standard image formats (e.g., JPEG, PNG)
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Failed to load the image.")
        return image

def save_image(image, file_path, bit_depth=8):
    """
    Save an image to a file in the specified bit depth.

    Args:
        image (numpy.ndarray): Image to save.
        file_path (str): Path to save the image.
        bit_depth (int): Target bit depth for the saved image.
    """
    image_writer = ImageWriter()
    image_writer.save_image(image, file_path, bit_depth)