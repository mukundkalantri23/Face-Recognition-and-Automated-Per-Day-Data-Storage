# This code will basically open up a camera window and would try to recognize all the faces present in the frame with the 
# help of the model trained last. For all the faces recognized, a box will appear around the face and the name of the 
# person identified would be showed under the box. That's how you know the faces are recognized correctly!
# A new text file is created for each day automatically (if it does not already exist), whenever the code is executed.
# This file will have the names of all those people who have been identified (atleast once) on that day, and so it 
# records the people who have been scanned on that day!

# ATTENTION:- Make sure you have a piece of code for all the users you want to be recognized! (See Below)

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
        
      # Write the code in the pattern as below, for all the users before running this script 
        
"""     if(id==1):
            id="Mukund";
        if(id==2):
            id="Tony";
        if(id==3):
            id="Thor";
                :
                :
                :            """

      # The names written here will be the names that appears on the screen when a face is recognized, so make sure you write them 
      # in front of the correct ID number!
        
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

# If you want more information to appear in the records (scanned faces file) like time or ID number, you can very well edit the code!
        
    cv2.imshow("face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()
