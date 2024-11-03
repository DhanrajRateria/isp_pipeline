from .stages import EdgeBasedDemosaic, GrayWorldWhiteBalance, GaussianDenoise, SRGBGamma, UnsharpMask

class ISPPipeline:
    def __init__(self):
        self.demosaic_stage = EdgeBasedDemosaic({})
        self.white_balance_stage = GrayWorldWhiteBalance({})
        self.denoise_stage = GaussianDenoise({})
        self.gamma_stage = SRGBGamma({})
        self.sharpen_stage = UnsharpMask({})

    def reset(self):
        pass

    def demosaic(self, image):
        return self.demosaic_stage.process(image)

    def white_balance(self, image, strength):
        return self.white_balance_stage.process(image, strength)

    def denoise(self, image, kernel_size):
        return self.denoise_stage.process(image, kernel_size)

    def gamma_correction(self, image, gamma):
        return self.gamma_stage.process(image, gamma)

    def sharpen(self, image, amount):
        return self.sharpen_stage.process(image, amount)