import cv2 as cv
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt


def rotate_and_cut_off(image: np.array, angle: float, center: (int, int)) -> np.array:
    height, width = image.shape[:2]
    x, y = center

    theta = angle / 180.0 * np.math.pi
    cos_t = np.math.cos(theta)
    sin_t = np.math.sin(theta)
    M = np.float32([[cos_t, sin_t, x - x * cos_t - y * sin_t], [-sin_t, cos_t, y + x * sin_t - y * cos_t]])

    new_width = int(-height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t - width * np.abs(sin_t))

    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y

    rotated = cv.warpAffine(image, M, (new_width, new_height))
    return rotated


def align_image(image: np.array, max_angle=5):
    height, width = image.shape[:2]

    max_variation = 0
    best_angle = None

    for angle in np.linspace(-max_angle, max_angle, 21):
        M = cv.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        rotated_img = cv.warpAffine(image, M, (width, height))

        x = [sum(1 - row / 255) for row in rotated_img]
        x_mean = sum(x) / len(x)
        x_RMSE = sqrt(sum((x - x_mean)**2) / len(x))
        x_variation = x_RMSE / x_mean

        if x_variation > max_variation:
            best_angle = angle
            max_variation = x_variation

    horizontal_img = rotate_and_cut_off(image, best_angle, (width // 2, height // 2))
    return horizontal_img[2:-2]


def cut_image_into_text_lines(img_path, valley_coef=0.04, slope_coef=0.02, deviation_bound=0.3, show_plot=False):
    image = cv.imread(img_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = align_image(image)

    height, width = image.shape[:2]
    valley_size = int(height * valley_coef)
    slope_size = int(height * slope_coef)

    x = [sum(1 - row / 255) for row in image]

    x_mean = sum(x) / len(x)
    x_RMSE = sqrt(sum((x - x_mean) ** 2) / len(x))

    good_indices = []
    for i in range(valley_size + slope_size, len(x) - valley_size):
        maybe_valley = x[i - valley_size: i]
        mean = sum(maybe_valley) / len(maybe_valley)
        derivation = (sum(x[i: i + slope_size]) - sum(x[i - slope_size: i])) / slope_size

        if mean < x_mean - x_RMSE / 2 and derivation / x_RMSE > deviation_bound:
            if show_plot:
                plt.plot([i, i], [0, 100], 'r-')
            good_indices.append(i)

    cut_indices = [0]
    for j in range(len(good_indices)):
        if j == 0 or good_indices[j] - good_indices[j - 1] > valley_size:
            if show_plot:
                plt.plot([good_indices[j], good_indices[j]], [0, 100], 'k-')
            cut_indices.append(good_indices[j])
    cut_indices.append(height)

    if show_plot:
        plt.plot([i for i in range(len(x))], x)
        plt.show()

    single_line_images = []
    for j in range(len(cut_indices) - 1):
        single_line_images.append(image[cut_indices[j]: cut_indices[j + 1]])

    return single_line_images


for j in range(1000):
    path = f'../../StandartImages/Train/DegradedImages/degraded_image{j}.png'
    print('Figurka' + str(j))
    for img in cut_image_into_text_lines(path):
        cv.imshow('img', img)
        cv.waitKey(0)


