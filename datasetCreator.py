#On running this code, the user would be asked to enter the ID number and then a camera window will open where a number of 
#pictures would be taken and processed to get the face in the picture. These pictures would be stored inside the dataSet folder
#with names according to the ID number provided and the number of picture it was.

#Recommended:- 
#1. while running this code, it is advisable that only one user should appear infront of the camera or else the dataset
#   generated would be faulty and so would be the training of the model and final recognizing of the person.
#2. The user may make different expressions and move their face a little so that the model could be trained with different 
#   variations of the users face.
#3. To improve the accuracy, the number of pictures taken per person (see below)  can be increased. (However it would add up space.)

import os
import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

cam=cv2.VideoCapture(0);

id=input('enter user id:- ')
sampleNum=0;
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100);
    cv2.imshow("face",img);
    cv2.waitKey(1);
    if(sampleNum>40): #Pictures of the user would be taken as long as this condition is true. You might alter this condition if you want.
        break
     

cam.release()
cv2.destroyAllWindows()
