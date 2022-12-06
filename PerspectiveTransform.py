import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class PerspectiveTransform():

    def __init__(self):
        self.videoPath = ''
        self.imagePath = ''

    def LoadVideo(self, fileName):
        self.videoPath = fileName

    def LoadImage(self, fileName):
        self.imagePath = fileName
    
    def DetectAndExecute(self):

        logo = cv2.imread(self.imagePath)
        capture = cv2.VideoCapture(self.videoPath)
        fps = capture.get(cv2.CAP_PROP_FPS)
        ret, frame = capture.read()

        while(capture.isOpened()):

            ret, frame = capture.read()
            if not ret:
                break

            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
            param = cv2.aruco.DetectorParameters_create()
            markerCornaers, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
                frame,
                dictionary,
                parameters = param
            )

            index1 = np.squeeze(np.where(markerIds == 1))
            index2= np.squeeze(np.where(markerIds == 2))
            index3 = np.squeeze(np.where(markerIds == 3))
            index4 = np.squeeze(np.where(markerIds == 4))

            if index1 != [] and index2 != [] and index3 != [] and index4 != []:

                refPt1 = np.squeeze(markerCornaers[index1[0]])[0]
                refPt2 = np.squeeze(markerCornaers[index2[0]])[1]
                refPt3 = np.squeeze(markerCornaers[index3[0]])[2]
                refPt4 = np.squeeze(markerCornaers[index4[0]])[3]

                pts_src = np.array([[0, 0], [logo.shape[1], 0], [logo.shape[1], 
                                logo.shape[0]],  [0, logo.shape[0]]], dtype=float)

                pts_dst = [[refPt1[0], refPt1[1]]]
                pts_dst = pts_dst + [[refPt2[0], refPt2[1]]]
                pts_dst = pts_dst + [[refPt3[0], refPt3[1]]]
                pts_dst = pts_dst + [[refPt4[0], refPt4[1]]]
                pts_dst = np.array(pts_dst)

                h, status = cv2.findHomography(pts_src, pts_dst)
                warped_image = cv2.warpPerspective(logo, h, (frame.shape[1], frame.shape[0]))
                cv2.fillConvexPoly(frame, pts_dst.astype(int), 0, 16)
                cv2.imshow('3.1 Perspective Transform', cv2.resize(frame+warped_image,(800,600)))

            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

        cv2.waitKey(0)
        capture.release()
        cv2.destroyAllWindows()

#參考 https://blog.csdn.net/LuohenYJ/article/details/105228916
