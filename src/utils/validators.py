import numpy as np

class InputValidator:
    """Validates input parameters for ISP pipeline stages."""

    @staticmethod
    def validate_image(image):
        """Validate input image array."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")

        if image.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (raw) or 3D (RGB)")

    @staticmethod
    def validate_kernel_size(size):
        """Validate kernel size for filters."""
        if not isinstance(size, int):
            raise TypeError("Kernel size must be an integer")

        if size < 3 or size % 2 == 0:
            raise ValueError("Kernel size must be odd and >= 3")