# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from BackgroundSubtraction import BackgroundSubstraction
from OpticalFlow import OpticalFlow
from PerspectiveTransform import PerspectiveTransform
from PCA import PCA


class Ui_MainWindow(object):

    def __init__(self):
        self.bgSub = BackgroundSubstraction()
        self.opticalFlow = OpticalFlow()
        self.perspTrans = PerspectiveTransform()
        self.pca = PCA()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(331, 647)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.loadVideo = QtWidgets.QPushButton(self.centralwidget)
        self.loadVideo.setGeometry(QtCore.QRect(40, 20, 251, 31))
        self.loadVideo.setObjectName("loadVideo")
        self.loadVideo.clicked.connect(self.LoadVideoClicked)

        self.loadImage = QtWidgets.QPushButton(self.centralwidget)
        self.loadImage.setGeometry(QtCore.QRect(40, 80, 251, 31))
        self.loadImage.setObjectName("loadImage")
        self.loadImage.clicked.connect(self.LoadImageClicked)

        self.loadFolder = QtWidgets.QPushButton(self.centralwidget)
        self.loadFolder.setGeometry(QtCore.QRect(40, 140, 251, 31))
        self.loadFolder.setObjectName("loadFolder")
        self.loadFolder.clicked.connect(self.LoadFolderClicked)

        self.videoTxt = QtWidgets.QLabel(self.centralwidget)
        self.videoTxt.setGeometry(QtCore.QRect(40, 60, 101, 16))
        self.videoTxt.setObjectName("videoTxt")
        self.imageTxt = QtWidgets.QLabel(self.centralwidget)
        self.imageTxt.setGeometry(QtCore.QRect(40, 120, 101, 16))
        self.imageTxt.setObjectName("imageTxt")
        self.folderTxt = QtWidgets.QLabel(self.centralwidget)
        self.folderTxt.setGeometry(QtCore.QRect(40, 180, 101, 16))
        self.folderTxt.setObjectName("folderTxt")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 200, 291, 61))
        self.groupBox.setObjectName("groupBox")

        self.backgroundSubtraction = QtWidgets.QPushButton(self.groupBox)
        self.backgroundSubtraction.setGeometry(QtCore.QRect(20, 20, 251, 31))
        self.backgroundSubtraction.setObjectName("backgroundSubtraction")
        self.backgroundSubtraction.clicked.connect(self.BackgroundSubtractionClicked)
        
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 280, 291, 101))
        self.groupBox_2.setObjectName("groupBox_2")
        self.preprocessing = QtWidgets.QPushButton(self.groupBox_2)
        self.preprocessing.setGeometry(QtCore.QRect(20, 20, 251, 31))
        self.preprocessing.setObjectName("preprocessing")
        self.preprocessing.clicked.connect(self.PreprocessingClicked)

        self.videoTracking = QtWidgets.QPushButton(self.groupBox_2)
        self.videoTracking.setGeometry(QtCore.QRect(20, 60, 251, 31))
        self.videoTracking.setObjectName("videoTracking")
        self.videoTracking.clicked.connect(self.VideoTrackingClicked)

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 400, 291, 61))
        self.groupBox_3.setObjectName("groupBox_3")

        self.perspectiveTransform = QtWidgets.QPushButton(self.groupBox_3)
        self.perspectiveTransform.setGeometry(QtCore.QRect(20, 20, 251, 31))
        self.perspectiveTransform.setObjectName("perspectiveTransform")
        self.perspectiveTransform.clicked.connect(self.PerspectiveTransformClicked)

        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(20, 480, 291, 101))
        self.groupBox_4.setObjectName("groupBox_4")

        self.imageReconstruction = QtWidgets.QPushButton(self.groupBox_4)
        self.imageReconstruction.setGeometry(QtCore.QRect(20, 20, 251, 31))
        self.imageReconstruction.setObjectName("imageReconstruction")
        self.imageReconstruction.clicked.connect(self.ImageReconstructionClicked)

        self.computeTheReconstructionError = QtWidgets.QPushButton(self.groupBox_4)
        self.computeTheReconstructionError.setGeometry(QtCore.QRect(20, 60, 251, 31))
        self.computeTheReconstructionError.setObjectName("computeTheReconstructionError")
        self.computeTheReconstructionError.clicked.connect(self.ComputeTheReconstructionErrorClicked)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 331, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadVideo.setText(_translate("MainWindow", "Load Video"))
        self.loadImage.setText(_translate("MainWindow", "Load Image"))
        self.loadFolder.setText(_translate("MainWindow", "Load Folder"))
        self.videoTxt.setText(_translate("MainWindow", "No video loaded"))
        self.imageTxt.setText(_translate("MainWindow", "No image loaded"))
        self.folderTxt.setText(_translate("MainWindow", "No folder loaded"))
        self.groupBox.setTitle(_translate("MainWindow", "1.Background Subtraction"))
        self.backgroundSubtraction.setText(_translate("MainWindow", "1.1 Background Subtraction"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Optical Flow"))
        self.preprocessing.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.videoTracking.setText(_translate("MainWindow", "2.2 Video Tracking"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Perspective Transform"))
        self.perspectiveTransform.setText(_translate("MainWindow", "3.1 Perspective Transform"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. PCA"))
        self.imageReconstruction.setText(_translate("MainWindow", "4.1 Image Reconstruction"))
        self.computeTheReconstructionError.setText(_translate("MainWindow", "4.2 Compute the reconstruction error"))

    def LoadVideoClicked(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile', './')
        if(fileName[0] != ''):
            self.videoTxt.setText('Video loaded')
            self.bgSub.LoadVideo(fileName[0])
            self.opticalFlow.LoadVideo(fileName[0])
            self.perspTrans.LoadVideo(fileName[0])
        else:
            self.videoTxt.setText('No video loaded')

    def LoadFolderClicked(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Load folder", "./")
        if(folder_path != ''):
            self.folderTxt.setText('Folder loaded')
            self.pca.LoadFolder(folder_path)
        else:
            self.folderTxt.setText('No folder loaded')

    def LoadImageClicked(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile', './')
        if(fileName[0] != ''):
            self.imageTxt.setText('Image loaded')
            self.perspTrans.LoadImage(fileName[0])
        else:
            self.imageTxt.setText('No image loaded')

    def BackgroundSubtractionClicked(self):
        self.bgSub.BackgroundSubtraction()

    def PreprocessingClicked(self):
        self.opticalFlow.Preprocessing()

    def VideoTrackingClicked(self):
        self.opticalFlow.VideoTracking()

    def PerspectiveTransformClicked(self):
        self.perspTrans.DetectAndExecute()

    def ImageReconstructionClicked(self):
        self.pca.ImageReconstruction()
    
    def ComputeTheReconstructionErrorClicked(self):
        self.pca.ComputeTheReconstructionError()

    