import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class BackgroundSubstraction():

    def __init__(self):
        self.videoPath = ''
    
    def LoadVideo(self, fileName):
        self.videoPath = fileName

    def BackgroundSubtraction(self):
        i = 0
        mean = []
        std = []
        frames = []

        capture = cv2.VideoCapture(self.videoPath)
        fps = capture.get(cv2.CAP_PROP_FPS)

        while(capture.isOpened()):

            ret, frame = capture.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)

            if i < 24:
                frames.append(gray)
            elif i == 24:
                frames.append(gray)
                frames = np.array(frames)
                mean = np.mean(frames, 0)
                std = np.std(frames, 0)
                std[std < 5] = 5
            else:
                diff = np.subtract(gray, mean)
                diff = np.absolute(diff)
                mask[diff > 5*std] = 255
                mask[diff <= 5*std] = 0

            result = cv2.bitwise_and(frame, frame, mask = mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            result = np.hstack((frame, mask, result))
            
            cv2.imshow('1.1 Background Subtraction', result)

            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

            i += 1

        capture.release()
        cv2.destroyAllWindows()
    
