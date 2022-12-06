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

        lk_params = dict(winSize = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        detector = cv2.SimpleBlobDetector_create(self.params)


        capture = cv2.VideoCapture(self.videoPath)
        fps = capture.get(cv2.CAP_PROP_FPS)

        ret, old_frame = capture.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(old_gray)
        p0 = np.array([[[kp.pt[0], kp.pt[1]]] for kp in keypoints]).astype(np.float32)
        mask = np.zeros_like(old_frame)

        while(capture.isOpened()):

            ret, frame = capture.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            good_new = p1[st==1]
            good_old = p0[st==1]

            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 255), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 255, 255), -1)
                
            img = cv2.add(frame, mask)
            
            cv2.imshow('2.2 Video tracking', img)

            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

        cv2.waitKey(0)
        capture.release()
        cv2.destroyAllWindows()

# 參考 https://reurl.cc/X5QjVD