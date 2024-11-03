import numpy as np
from .stages import demosaic, white_balance, denoise, gamma, sharpen
from .utils.image_io import read_raw_image, save_processed_image

class ISPPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.stages = []
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Initialize processing stages with configuration"""
        self.stages = [
            ("demosaic", demosaic.EdgeBasedDemosaic(self.config.get("demosaic", {}))),
            ("white_balance", white_balance.GrayWorld(self.config.get("white_balance", {}))),
            ("denoise", denoise.GaussianDenoise(self.config.get("denoise", {}))),
            ("gamma", gamma.SRGBGamma(self.config.get("gamma", {}))),
            ("sharpen", sharpen.UnsharpMask(self.config.get("sharpen", {})))
        ]

    def process_image(self, raw_image, active_stages=None):
        """
        Process raw image through the pipeline

        Args:
            raw_image: Input 12-bit Bayer raw image
            active_stages: List of stages to apply (for comparison purposes)

        Returns:
            Processed image as 8-bit RGB
        """
        img = raw_image.copy()
        processed_stages = []

        for stage_name, processor in self.stages:
            if active_stages is None or stage_name in active_stages:
                img = processor.process(img)
                processed_stages.append(stage_name)

        return img, processed_stages