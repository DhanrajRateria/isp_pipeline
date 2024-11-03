import cv2
from .base import PipelineStage

class GaussianDenoise(PipelineStage):
    """Gaussian denoising using a 5x5 Gaussian filter."""
    def process(self, image):
        self.validate_input(image)
        return cv2.GaussianBlur(image, (5, 5), self.config.get('sigma', 1.0))
