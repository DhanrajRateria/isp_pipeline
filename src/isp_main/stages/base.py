from abc import ABC, abstractmethod
import numpy as np
import cv2

class PipelineStage(ABC):
    """Base class for all pipeline stages."""
    def __init__(self, config=None):
        self.config = config or {}

    @abstractmethod
    def process(self, image):
        """Process the input image."""
        pass

    def validate_input(self, image):
        """Basic input validation to check for numpy array format."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array.")