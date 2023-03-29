import sys
import cv2
import numpy as np
import os
import shutil
from random import shuffle


BACKGROUND = [0, 0, 0][::-1]
ITEM_RGHT = [255, 30, 30][::-1]
ITEM_LEFT = [30, 30, 255][::-1]
ITEM_BOTH = [30, 200, 255][::-1]
RIGHT_WRIST = [255, 30, 255][::-1]
LEFT_WRIST = [255, 130, 30][::-1]
RIGHT_WRIST_ELBOW = [140, 30, 255][::-1]
LEFT_WRIST_ELBOW = [255, 255, 30][::-1]
RIGHT_ELBOW_SHOULDER = [30, 120, 255][::-1]
LEFT_ELBOW_SHOULDER = [30, 255, 30][::-1]
FOREGROUND = [195, 195, 195]

def analyze_zero10_mask(img):
    has_bad_colour = False
    output_img = np.zeros_like(img)

    background = np.all(img==BACKGROUND, axis=2)
    item_right = np.all(img == ITEM_RGHT, axis=2)
    item_left = np.all(img == ITEM_LEFT, axis=2)
    item_both = np.all(img == ITEM_BOTH, axis=2)
    right_wrist = np.all(img == RIGHT_WRIST, axis=2)
    left_wrist = np.all(img == LEFT_WRIST, axis=2)
    right_wrist_elbow = np.all(img == RIGHT_WRIST_ELBOW, axis=2)
    left_wrist_elbow = np.all(img == LEFT_WRIST_ELBOW, axis=2)
    right_elbow_shoulder = np.all(img == RIGHT_ELBOW_SHOULDER, axis=2)
    left_elbow_shoulder = np.all(img == LEFT_ELBOW_SHOULDER, axis=2)
    foreground = np.all(img==FOREGROUND, axis=2)

    output_img[~(background | item_right | item_left | item_both | right_wrist | left_wrist | right_wrist_elbow | left_wrist_elbow | right_elbow_shoulder | left_elbow_shoulder | foreground)] = [255, 255, 255]
    has_bad_colour = np.sum(np.all(output_img!=[0,0,0], axis=2)) > 0

    return has_bad_colour, output_img