import pytesseract as tesseract
from exceptions import UnableToLoadModel
from base import OCRModel
import numpy as np
import cv2 as cv


class TESSERACT(OCRModel):
    SAVED_MODEL_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def load(self):
        if not self._loaded:
            try:
                tesseract.pytesseract.tesseract_cmd = self.SAVED_MODEL_PATH
                self._model = tesseract.image_to_string
            except Exception as ex:
                raise UnableToLoadModel(ex)
            else:
                self._loaded = True

    @staticmethod
    def preprocess_image(image: np.array) -> np.array:
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return grayscale_image
