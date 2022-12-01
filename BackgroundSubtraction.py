import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class BackgroundSubstraction():

    def __init__(self):
        self.videoPath = ''
    
    def LoadVideo(self, fileName):
        self.videoPath = fileName
        print(self.videoPath)

    def BackgroundSubtraction(self):
        pass
    
