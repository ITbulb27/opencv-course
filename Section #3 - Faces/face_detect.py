#pylint:disable=no-member

import cv2 as cv
import os

image_path = '../Resources/Photos/naruto.jpg'
cascade_path = 'naruto1.xml'

# 1. Safety Check para sa Image File
if not os.path.exists(image_path):
    print(f"Error: Ang image file wala makit-i sa: '{image_path}'")
    print("Palihug i-tsek ang imong folder structure ug ang pangalan sa file.")

# 2. Safety Check para sa Haar Cascade XML File
elif not os.path.exists(cascade_path):
    print(f"Error: Ang XML file wala makit-i sa: '{cascade_path}'")
    print("Siguroha nga naa ang 'naruto 1.xml' sa parehas nga folder diin nimo gipadagan kini nga script.")

else:
    # Basahon ang image kung naa ang duha ka file
    img = cv.imread(image_path)
    cv.imshow('Group of 5 people', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray People', gray)

    # I-load ang cascade classifier
    haar_cascade = cv.CascadeClassifier(cascade_path)

    # Pag-detect sa mga nawong (Giusab ang minNeighbors ngadto sa 3 para mas accurate)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    print(f'Number of faces found = {len(faces_rect)}')

    # Pag-draw og rectangle sa matag nawong
    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv.imshow('Detected Faces', img)

    # Kani magpugong nga dili mo-close ang mga windows
    cv.waitKey(0)
    cv.destroyAllWindows()
