import numpy as np
from .base import PipelineStage

class EdgeBasedDemosaic(PipelineStage):
    """5x5 edge-based demosaicing for GRBG Bayer pattern images."""
    def process(self, raw_image):
        h, w = raw_image.shape
        output = np.zeros((h, w, 3), dtype=np.float32)

        # Green channel interpolation
        green_mask = np.zeros((h, w), dtype=bool)
        green_mask[0::2, 0::2] = True  # G positions in GRBG
        green_mask[1::2, 1::2] = True  # G positions in GRBG
        output[:, :, 1][green_mask] = raw_image[green_mask]

        # Red and blue channel interpolation
        for i in range(2, h-2):
            for j in range(2, w-2):
                # Red channel (top-left of each 2x2 GRBG block)
                if i % 2 == 0 and j % 2 == 0:
                    output[i, j, 0] = raw_image[i, j]
                    output[i+1, j+1, 0] = np.mean([raw_image[i, j], raw_image[i+2, j+2]])
                # Blue channel (bottom-right of each 2x2 GRBG block)
                elif i % 2 == 1 and j % 2 == 1:
                    output[i, j, 2] = raw_image[i, j]
                    output[i-1, j-1, 2] = np.mean([raw_image[i, j], raw_image[i-2, j-2]])

        return output