import cv2 as cv
from tracker import *

#Tracker Object
tracker=EuclideanDistTracker()

cap=cv.VideoCapture("videoplayback.mp4")

#Object detection from Stable Camera
object_detector=cv.createBackgroundSubtractorMOG2(history=100,varThreshold=100)
#higher the history, more precise the value
#lower the threshold, more precise the detection

while True:
    ret,frame=cap.read()

    #putting region of interest
    roi=frame[100:400,100:600]

    #Mask
    mask=object_detector.apply(roi)
    _,mask=cv.threshold(mask,254,255,cv.THRESH_BINARY)

    contours,_=cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    #the _ above is for heirarchy

    detections=[]
    for cnt in contours:
        #removing small areas
        area=cv.contourArea(cnt)
        if area>100:
            #cv.drawContours(roi,[cnt],-1,(0,0,255),2)
            x,y,w,h=cv.boundingRect(cnt)
            cv.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),3)

            detections.append([x,y,w,h])



    #Object Tracking
    box_ids=tracker.update(detections)
    for box_id in box_ids:
        x,y,w,h,id=box_id
        #cv.putText(roi,str(id),(x,y-15),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
        cv.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),3)

    #print(detections)
    cv.imshow("Frame",frame)
    #cv.imshow("roi",roi)
    #cv.imshow("Mask",mask)

    key = cv.waitKey(30)
    if key==27: 
        break

cap.release()
cv.destroyAllWindows()