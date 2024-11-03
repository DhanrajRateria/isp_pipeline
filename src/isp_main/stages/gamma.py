import numpy as np
from .base import PipelineStage

class SRGBGamma(PipelineStage):
    """sRGB gamma correction."""
    def process(self, image):
        self.validate_input(image)
        normalized = image.astype(np.float32) / image.max()  # Normalize to [0, 1]
        corrected = np.power(normalized, 1 / self.config.get('gamma', 2.2))
        return (corrected * 255).clip(0, 255).astype(np.uint8)  # Convert to 8-bit
