import cv2
import numpy
import face_recognition
import os
from datetime import datetime

import numpy as np

path = 'attendeeimage'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
    print(classnames)

def findencodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
def attendence(name):
    with open('attendence.csv','r+') as f:
        MyDatalist = f.readlines()
        nameList = []

        for line in MyDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            datestring  = now.strftime('%H %M %S')
            f.writelines(f'\n{name},{datestring}')



encodeListKnown = findencodings(images)
print('Encoding Complete')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceFrame = face_recognition.face_locations(imgS)
    encodeFrame = face_recognition.face_encodings(imgS,faceFrame)

    for encodeFace,faceloc in zip(encodeFrame,faceFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)
            y1,y2,x2,x1 = faceloc
            y1, y2, x2, x1 = y1*4, y2*4, x2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
            attendence(name)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

