import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as decom

class PCA():

    def __init__(self):
        self.folderPath = ''
        self.imgPath = []
        self.orgImg = []
        self.newImg = []
        self.err = []
    def LoadFolder(self, path):
        self.folderPath = path
        self.imgPath = os.listdir(self.folderPath)
        self.imgPath.sort(key=lambda x:int(x[8:-5]))
    
    def ImageReconstruction(self):
        
        for filename in self.imgPath:
            img = cv2.imread(self.folderPath + '/' + filename)
            self.orgImg.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            b, g, r = cv2.split(img)

            pca = decom.PCA(n_components= 5)

            bNew = pca.inverse_transform(pca.fit_transform(b))
            gNew = pca.inverse_transform(pca.fit_transform(g))
            rNew = pca.inverse_transform(pca.fit_transform(r))
        
            bNew = np.clip(bNew, a_min = 0, a_max = 255)
            gNew = np.clip(gNew, a_min = 0, a_max = 255)
            rNew = np.clip(rNew, a_min = 0, a_max = 255)

            self.newImg.append(cv2.cvtColor((cv2.merge([bNew, gNew, rNew])).astype(np.uint8), cv2.COLOR_BGR2RGB))

        fig = plt.figure(figsize=(15, 4))

        for i in range(15):
            plt.subplot(4, 15, i+1)
            plt.axis('off')
            plt.imshow(self.orgImg[i])

            plt.subplot(4, 15, i + 16)
            plt.axis('off')
            plt.imshow(self.newImg[i])
        
        for i in range(15,30):
            plt.subplot(4, 15, i+16)
            plt.axis('off')
            plt.imshow(self.orgImg[i])

            plt.subplot(4, 15, i + 31)
            plt.axis('off')
            plt.imshow(self.newImg[i])

        fig.text(0, 0.9, 'origin', verticalalignment='center', rotation='vertical', horizontalalignment = 'left')
        fig.text(0, 0.65, 'reconstruction', verticalalignment='center', rotation='vertical', horizontalalignment = 'left')
        fig.text(0, 0.4, 'origin', verticalalignment='center', rotation='vertical', horizontalalignment = 'left')
        fig.text(0, 0.15, 'reconstruction', verticalalignment='center', rotation='vertical', horizontalalignment = 'left')

        plt.tight_layout(pad=1)
        plt.show()
        


