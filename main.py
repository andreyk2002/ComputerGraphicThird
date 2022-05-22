# Реализация поэлементных операций + линейное контрастирование
# Морфологическая обработка
import math

import cv2
import numpy as np
import os
import matplotlib.pyplot as plot


def add_constant(original_image, value: int):
    image = original_image.copy()
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            image[x, y] += value
    print('Оригинал vs Изображение с добавлением константы=' + str(value))
    f, (ax1, ax2) = plot.subplots(1, 2, sharey=True, figsize=(10, 10))
    ax1.imshow(original_image)
    ax2.imshow(image)
    plot.show()


def do_negative(original_image):
    image = original_image.copy()
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            image[x, y] = 255 - image[x, y]
    print('Оригинал vs Негатив')
    f, (ax1, ax2) = plot.subplots(1, 2, sharey=True, figsize=(10, 10))
    ax1.imshow(original_image)
    ax2.imshow(image)
    plot.show()


def multiply(original_image, value: float):
    image = original_image.copy()
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            image[x, y] = int(value * image[x, y])
    print('Оригинал vs Изображение с умножением на константу=' + str(value))
    f, (ax1, ax2) = plot.subplots(1, 2, sharey=True, figsize=(10, 10))
    ax1.imshow(original_image)
    ax2.imshow(image)
    plot.show()


def pow(original_image, value: float):
    image = original_image.copy()
    max_f = np.matrix(original_image).max()
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            image[x, y] = int(255 * (image[x, y] / max_f) ** value)
    print('Оригинал vs Изображение в степени=' + str(value))
    f, (ax1, ax2) = plot.subplots(1, 2, sharey=True, figsize=(10, 10))
    ax1.imshow(original_image)
    ax2.imshow(image)
    plot.show()


def log(original_image):
    image = original_image.copy()
    max_f = np.matrix(original_image).max()
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            image[x, y] = int(255 * (math.log(1 + image[x, y]) / math.log(1 + max_f)))
    print('Оригинал vs Логарифм')
    f, (ax1, ax2) = plot.subplots(1, 2, sharey=True, figsize=(10, 10))
    ax1.imshow(original_image)
    ax2.imshow(image)
    plot.show()


def linear_contr(original_image):
    image = original_image.copy()
    max_f = np.matrix(original_image).max()
    min_f = np.matrix(original_image).min()
    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            image[x, y] = int(255 / (max_f - min_f) * (image[x, y] - min_f))
    print('Оригинал vs Линейное контрастирование')
    f, (ax1, ax2) = plot.subplots(1, 2, sharey=True, figsize=(10, 10))
    ax1.imshow(original_image)
    ax2.imshow(image)
    plot.show()


def morphological_processing(original_image, structural_element: str):
    kernel = np.array([1, 1, 1])
    if structural_element == 'central':
        kernel = np.ones((3, 3), np.uint8)
    elif structural_element == 'vertical':
        kernel = np.array([[1], [1], [1]])
    eroded = original_image.copy()
    cv2.erode(src=eroded, kernel=kernel)
    dilateted = original_image.copy()
    cv2.dilate(src=dilateted, kernel=kernel)
    opened = original_image.copy()
    cv2.dilate(cv2.erode(src=opened, kernel=kernel), kernel=kernel)
    closed = original_image.copy()
    cv2.erode(cv2.dilate(src=closed, kernel=kernel), kernel=kernel)
    f, (ax1, ax2, ax3, ax4, ax5) = plot.subplots(1, 5, sharey=True, figsize=(20, 20))
    ax1.imshow(original_image)
    ax2.imshow(eroded)
    ax3.imshow(dilateted)
    ax4.imshow(opened)
    ax5.imshow(closed)
    plot.show()




# def dilation():
#
# def opening():
#
# def closing():

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_dir = 'test/'
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            input_image = cv2.imread(img_dir + file)
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            add_constant(gray_image, 13)
            multiply(gray_image, 1.2)
            pow(gray_image, 0.5)
            log(gray_image)
            linear_contr(gray_image)
            (thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            structural = input('Chose structural element type = central, cross, horizontal, vertical')
            morphological_processing(blackAndWhiteImage, structural)
