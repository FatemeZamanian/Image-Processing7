import cv2
import random
import numpy as np
import argparse
from mtcnn import MTCNN
import math
from PIL import Image

import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()
image=cv2.imread(args.image)

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1- prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def rotate(im):
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(im)
    detection = results[0]
    keypoints = detection["keypoints"]
    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]
    #################################
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    if left_eye_y > right_eye_y:
        point3=(right_eye_x, left_eye_y)
        point3_x = right_eye_x
        point3_y=left_eye_y
        direction = -1
    else:
        point3=(left_eye_x, right_eye_y)
        point3_x = left_eye_x
        point3_y = right_eye_y
        direction = 1

    a = math.sqrt(math.pow(left_eye_x-point3_x,2) + math.pow(left_eye_y-point3_y,2))
    b = math.sqrt(math.pow(right_eye_x - point3_x, 2)+math.pow(right_eye_y - point3_y, 2))
    c = math.sqrt(math.pow(left_eye_x - right_eye_x, 2)+math.pow(left_eye_y - right_eye_y, 2))
    cos=b/c

    angle = np.arccos(cos)
    angle = (angle * 180) / math.pi
    if direction == -1:
        angle = 90 - angle

    im=Image.fromarray(im)
    result=np.array(im.rotate(direction * angle))
    result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    return result

# image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
noisy_bean= sp_noise(image,0.05)
# cv2.imshow('',noisy_bean)
clean_bean=cv2.medianBlur(noisy_bean,3)
# cv2.imshow('',clean_bean)
rotation=rotate(image)
cv2.imshow('',rotation)
cv2.waitKey()