import os
import numpy as np
from PIL import Image
import cv2


recog = cv2.face.LBPHFaceRecognizer_create()
path = '/'

def getImagewithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []

    for imgpath in imagePaths:
        faceImg = Image.open(imgpath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imgpath)[-1].split('_')[0])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    
    return IDs , faces

IDs,faces = getImagewithID(path)
recog.train(faces,np.array(IDs))
recog.write('C:/Users/Shubham/IOSD-ML-master/train.yml')
cv2.destroyAllWindows
