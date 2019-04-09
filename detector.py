import cv2
import os
import numpy as np
import datetime

now = datetime.datetime.now()
dat = str(now.day)+"-"+str(now.month)+"-"+str(now.year)
filename = "scannedfaces "+dat+".txt"
exists = os.path.isfile(filename)
if not exists:
    f = open(filename,"w")
    intro = "This is file containing names of people scanned on " + dat + "\n\n"
    f.write(intro)
    f.close()

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
print("Press 'q' to exit the camera window.")
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="mukund";
        if(id==2):
            id="rdj";
        if(id==3):
            id="mudit";
        if(id==4):
            id="madhav";
        cv2.putText(img,str(id),(x,y+h),font,2,(0,0,255),2);
        id = id + "\n"
        f = open(filename,"r")
        namestatus = "no"
        for count,name in enumerate(f,1):
            if(name == id):
                namestatus = "yes"
        if(namestatus == "no"):
            f = open(filename,"a")
            f.write(id)

    cv2.imshow("face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()
