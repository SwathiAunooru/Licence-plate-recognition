# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:22:04 2019
@author: swathi
"""
import cv2
import pytesseract
import dlib

#Cascade Files for detecting cars in the region
car_cascade = cv2.CascadeClassifier('cascade_cust_trail3.xml')

cap =  cv2.VideoCapture("test2.avi")
#cap =  cv2.VideoCapture(0)

print("-----------------------Video frame grabbed----------------------------")
i = 0
while True:
    _,image = cap.read()
    
    if (type(image) == type(None)):
        break
        
    #Detecting cars in a video
    cars = car_cascade.detectMultiScale(image, 2, 3) #(image,objects,scaleFactor,minNeighbors,flags,minSize,maxSize)
#    tracker = dlib.correlation_tracker()
#    print(tracker)
    for (x,y,w,h) in cars:
#        tracker = dlib.correlation_tracker()
#        tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)     
        detect = image[y:y+h,x:x+w]     
        print("---------------------------VLP--------------------------------")
        #Increasing the contrast of image
        alpha = 1.5 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        detect = cv2.convertScaleAbs(detect, alpha=alpha, beta=beta)
        detect = cv2.fastNlMeansDenoisingColored(detect, None, 10, 10, 7, 15) 

        scale_percent = 220 # percent of original size
        width = int(detect.shape[1] * scale_percent / 100)
        height = int(detect.shape[0] * scale_percent / 100)
        dim = (width, height)

        #resized image
        detect = cv2.resize(detect, dim, interpolation = cv2.INTER_AREA)
        
        print("--------text here-------------starts")
        gray = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Final detection",gray)
        
        if True:
            j = "Croppeda"+str(i)+".jpg"
#            cv2.imwrite(j,gray)
            i = i+1
            print(i)
        text = pytesseract.image_to_string(gray)
        print("---------------printing vehicle number plate------------------") 
        print(text)
  
    cv2.imshow('video', image)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()