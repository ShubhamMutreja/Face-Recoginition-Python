import cv2
import numpy as numpy

face = cv2.CascadeClassifier('C:/Users/Shubham/IOSD-ML-master/haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('C:/Users/Shubham/IOSD-ML-master/train.yml')

id = 0

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,255,255)

id_map = ['Shubham']

cam = cv2.VideoCapture(0)

while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id , conf = rec.predict(gray[y:y+h])

        cv2.putText(img,str(id_map[id-1])+"_"+str(conf),(x,y+h),fontFace,fontScale, fontColor)
    
    cv2.imshow("face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()