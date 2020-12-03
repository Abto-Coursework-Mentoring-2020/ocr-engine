import numpy as np
from PIL import Image
import cv2 as cv
from tesseract import TESSERACT
import json
from metrics import edit_distance


def take_clear_image_text(input_dir, clear_image_name):
    res = ''
    with open(input_dir) as json_file:
        file = json.load(json_file)
        clear_image_data = file[clear_image_name]

        for word_data in clear_image_data:
            word = str(word_data['word'])
            if word.find('\n') == -1:
                res += word + ' '
            else:
                res += word
    res += '\f'
    return res


'''
with open("../../DegradedImages/word_coordinates.json") as json_file:
    file_degraded = json.load(json_file)
    
ans = 0
for i in range(10000):
    degraded_image_name = f'degraded_image{i}.png'
    degraded_image_data = file_degraded[degraded_image_name]

    mis = degraded_image_data['tesseract_relative_mistake']
    ans += mis

print(ans/10000)
'''


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

