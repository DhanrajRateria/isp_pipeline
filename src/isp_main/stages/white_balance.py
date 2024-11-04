import numpy as np
from .base import PipelineStage

class GrayWorldWhiteBalance(PipelineStage):
    """Gray World white balance to adjust color balance."""
    def process(self, image, params = None):
        self.validate_input(image)
        r_avg, g_avg, b_avg = [np.mean(image[:, :, i]) for i in range(3)]
        gray = (r_avg + g_avg + b_avg) / 3
        scales = [gray / channel_avg if channel_avg > 0 else 1 for channel_avg in [r_avg, g_avg, b_avg]]
        result = image.copy()
        for i, scale in enumerate(scales):
            result[:, :, i] = np.clip(image[:, :, i] * scale, 0, 255)
        return result.astype(np.uint8)
