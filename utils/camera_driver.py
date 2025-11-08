from controller import Camera
import cv2
import numpy as np


def read_img(camera: Camera):
    width = camera.getWidth()
    height = camera.getHeight()
    img_bytes = camera.getImage()
    img = np.frombuffer(img_bytes, np.uint8).reshape((height, width, 4)).copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img
