import easyocr
from exceptions import UnableToLoadModel
from base import OCRModel
import numpy as np
import cv2 as cv


class EasyOCR(OCRModel):

    def load(self):
        if not self._loaded:
            try:
                self._model = easyocr.Reader(['en']).readtext
            except Exception as ex:
                raise UnableToLoadModel(ex)
            else:
                self._loaded = True

    @staticmethod
    def preprocess_image(image: np.array) -> np.array:
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return grayscale_image
