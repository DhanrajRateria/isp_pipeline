import numpy as np
import cv2

class ImageMetrics:
    """Calculates various image quality metrics."""

    @staticmethod
    def calculate_snr(image, region=None):
        """Calculate Signal-to-Noise Ratio."""
        if region is None:
            region = image

        signal = np.mean(region)
        noise = np.std(region)
        return 20 * np.log10(signal / noise) if noise != 0 else float('inf')

    @staticmethod
    def calculate_psnr(original, processed):
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    @staticmethod
    def calculate_sharpness(image):
        """Calculate image sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)