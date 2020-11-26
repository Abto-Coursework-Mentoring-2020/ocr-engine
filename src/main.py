import numpy as np
from PIL import Image
import cv2 as cv
from tesseract import TESSERACT
from metrics import edit_distance


def save_image(image: np.array, filename):
    img = Image.fromarray(np.uint8(image))
    img.save(filename)


def show_image(image: np.array):
    cv.imshow('img', image)
    cv.waitKey(0)


image1_path = "../sample_images/clear_image0.png"
image2_path = "../sample_images/clear_image1.png"
image1 = cv.imread(image1_path)
image2 = cv.imread(image2_path)
images = [image1, image2]

model = TESSERACT()
model.load()

for text in model.recognize_text(images):
    print(text)

