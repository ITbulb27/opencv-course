#pylint:disable=no-member

import cv2 as cv
import numpy as np
import os

image_path = '../Resources/Photos/pusa.jpg'

# 1. Safety check to prevent NoneType errors if path is wrong
if not os.path.exists(image_path):
    print(f"Error: The image file was not found at '{image_path}'")
    print("Please verify your folder structure and filename.")
else:
    img = cv.imread(image_path)
    cv.imshow('Pusa', img)

    # 2. Dimensions setup
    height, width = img.shape[:2]

    # 3. Create the blank mask canvas (must be 2D single-channel grayscale)
    blank = np.zeros((height, width), dtype='uint8')
    cv.imshow('Blank Image', blank)

    # 4. Create the circle mask using safe bounds
    circle_center = (width // 2 + 45, height // 2)
    circle = cv.circle(blank.copy(), circle_center, 100, 255, -1)

    # 5. Create the rectangle mask using safe bounds to avoid going outside the image
    max_x = min(370, width)
    max_y = min(370, height)
    rectangle = cv.rectangle(blank.copy(), (30, 30), (max_x, max_y), 255, -1)

    # 6. Combine the shapes
    weird_shape = cv.bitwise_and(circle, rectangle)
    cv.imshow('Weird Shape', weird_shape)

    # 7. Apply the mask onto the 3-channel color image
    masked = cv.bitwise_and(img, img, mask=weird_shape)
    cv.imshow('Weird Shaped Masked Image', masked)

    cv.waitKey(0)
    cv.destroyAllWindows()
