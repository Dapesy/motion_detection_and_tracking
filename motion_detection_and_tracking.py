import cv2 as cv
import numpy as np
cap = cv.VideoCapture('data/opencv-4.x/opencv/samples/data/vtest.avi')
# cap = cv.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv.absdiff(frame1, frame2)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blurr = cv.GaussianBlur(gray, (5,5), 0)
    _, threshold = cv.threshold(blurr, 20, 255, cv.THRESH_BINARY)
    dilate = cv.dilate(threshold, None, iterations=3)
    contours, hierarchy = cv.findContours(dilate , cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x,y,w,h) = cv.boundingRect(contour)
        if cv.contourArea(contour) < 800:
            continue
        cv.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0),2)
        cv.putText(frame1, "status : {}".format('movement'), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 3)

    # cv.drawContours(frame1, contours, -1, (0,255,0),2)

    cv.imshow('feed', frame1)

    # to get the different between the frames 

    frame1 = frame2
    ret , frame2 = cap.read()
    # cv.imshow('feed1', frame1)

    if cv.waitKey(40) == 27:
        break

cv.destroyAllWindows
cap.release()