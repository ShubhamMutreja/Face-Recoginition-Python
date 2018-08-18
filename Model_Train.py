import cv2
import numpy as np 

cam = cv2.VideoCapture(0)  
face = cv2.CascadeClassifier('C:/Users/Shubham/IOSD-ML-master/haarcascade_frontalface_default.xml')

uid = input('Enter Bish Name')

sampleNum = 0

while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in faces:
        sampleNum+=1
        cv2.imwrite('C:/Users/Shubham/IOSD-ML-master/Data/'+str(uid) + '_' + str(sampleNum) + '.jpg' , gray[y:y+h , x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)


    cv2.imshow("face",img)
    cv2.waitKey(1)
    if(sampleNum>100):
        break


cam.release()
cv2.destroyAllWindows()

