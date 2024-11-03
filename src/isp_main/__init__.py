from .stages.demosaic import EdgeBasedDemosaic
from .stages.white_balance import GrayWorldWhiteBalance
from .stages.denoise import GaussianDenoise
from .stages.gamma import SRGBGamma
from .stages.sharpen import UnsharpMask
from .pipeline import ISPPipeline
from .stages import demosaic, denoise, gamma, sharpen, white_balance

class ISPPipeline:
    def __init__(self):
        self.demosaic_stage = EdgeBasedDemosaic({})
        self.white_balance_stage = GrayWorldWhiteBalance({})
        self.denoise_stage = GaussianDenoise({})
        self.gamma_stage = SRGBGamma({})
        self.sharpen_stage = UnsharpMask({})

    def reset(self):
        # Optionally reset any pipeline state if needed
        pass

    def demosaic(self, image):
        return self.demosaic_stage.process(image)

    def white_balance(self, image, strength):
        self.white_balance_stage.config['strength'] = strength
        return self.white_balance_stage.process(image)

    def denoise(self, image, kernel_size):
        self.denoise_stage.config['kernel_size'] = kernel_size
        return self.denoise_stage.process(image)

    def gamma_correction(self, image, gamma):
        self.gamma_stage.config['gamma'] = gamma
        return self.gamma_stage.process(image)

    def sharpen(self, image, amount):
        self.sharpen_stage.config['amount'] = amount
        return self.sharpen_stage.process(image)
