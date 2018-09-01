#!/usr/bin/python
import math
import numpy as np
import cv2
from PIL import Image
import pytesseract
import os
import imutils
#dictionary of all contours
contours = {}
#array of edges of polygon
approx = []
#scale of the text
scale = 1
#camera
cap = cv2.VideoCapture(0)
cap.set(3, 280)
cap.set(4, 280)
print("press q to exit")

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (150,150))


while(cap.isOpened()):
    #Capture frame-by-frame
    ret, frame = cap.read()
    if ret==True:
        #grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Canny
        canny = cv2.Canny(frame,80,240,3)

        #contours
        canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0,len(contours)):
            #approximate the contour with accuracy proportional to
            #the contour perimeter
            approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

            #Skip small or non-convex objects
            if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
                continue

            vtc = len(approx)
                #to determine the shape of the contour
            x,y,w,h = cv2.boundingRect(contours[i])
            if(vtc==4):
                    
                cv2.putText(frame,'RECT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
#                crop_img = frame[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
                cv2.imwrite("face.jpeg",frame)
#                final_img = cv2.imread("face.jpeg")
#                final_img = imutils.resize(final_img,            

        #Display the resulting frame
#        out.write(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('canny',canny)
        if cv2.waitKey(1) == 1048689: #if q is pressed
            break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
imgg = cv2.imread("face.peg")
gray1 = cv2.cvtColor(imgg,cv2.COLOR_BGR2GRAY)
gray1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)[1]
gray1 = cv2.medianBlur(gray,3)
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename,gray1)
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
