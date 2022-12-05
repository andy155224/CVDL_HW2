import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class OpticalFlow():

    def __init__(self):
        self.videoPath = ''
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.minArea = 35
        self.params.maxArea = 90
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.8
        self.params.maxCircularity = 1
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.5
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.5

    def LoadVideo(self, fileName):
        self.videoPath = fileName

    def Preprocessing(self):
        capture = cv2.VideoCapture(self.videoPath)
        ret, img = capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.SimpleBlobDetector_create(self.params)
        keypoints = detector.detect(gray)

        for kp in keypoints:
            x, y = map(lambda x: int(x), kp.pt)
            img = cv2.rectangle(img, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 1)
            img = cv2.line(img, (x, y - 6), (x, y + 6), (0, 0, 255), 1)
            img = cv2.line(img, (x - 6, y), (x + 6, y), (0, 0, 255), 1)

        cv2.imshow("Circle detect", img)
        cv2.waitKey(0)
        capture.release()
        cv2.destroyAllWindows()

    def VideoTracking(self):
        pass