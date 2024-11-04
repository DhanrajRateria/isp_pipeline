import cv2
import numpy as np
from .base import PipelineStage

class UnsharpMask(PipelineStage):
    """Unsharp masking for enhancing image details."""
    def process(self, image, params = None):
        self.validate_input(image)
        blurred = cv2.GaussianBlur(image, (self.config.get('radius', 1)*2+1,)*2, self.config.get('radius', 1))
        mask = image - blurred
        sharpened = image + self.config.get('amount', 1.0) * mask
        return np.clip(sharpened, 0, 255).astype(np.uint8)