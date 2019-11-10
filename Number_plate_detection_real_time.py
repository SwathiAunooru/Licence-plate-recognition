# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:57:51 2019
@author: swathi
"""
import cv2
import pytesseract

#Cascade Files for detecting cars in the region
car_cascade = cv2.CascadeClassifier('cascade_cust_trail3.xml')

cap =  cv2.VideoCapture("test2.avi")
#cap =  cv2.VideoCapture(0)
print("----------------------------Video frame grabbed-------------------------------")
while True:
    _,image = cap.read()
    
    if (type(image) == type(None)):
        break
#    xx = 317; yy = 173; ww = 330; hh = 172; # for 2way video
    xx = 5; yy = 236; ww = 1515; hh = 423; # for dubai video
    #Croping the image using Roi
    imcrop=image[yy:yy+hh,xx:xx+ww]
    cv2.imwrite('IMCROP.jpg',imcrop)
    #Drawing Rectangle across ROI of an image
    cv2.rectangle(image,(xx,yy),(xx+ww,yy+hh),(255,255,255),2)
#    cv2.imshow("Rectangle on whole frame",image)
        
    #Detecting cars in a video
    cars = car_cascade.detectMultiScale(imcrop,5, 2, 3,(4,5))  #with roi region
#    cars = car_cascade.detectMultiScale(image, 2, 3)   #without ROI region
    
    for (x,y,w,h) in cars:
        
#        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)     #without ROI region
#        detect = image[y:y+h,x:x+w]            #without ROI region
        
        
        cv2.rectangle(imcrop,(x,y),(x+w,y+h),(0,255,255),2)      #with roi region
        detect = imcrop[y:y+h,x:x+w]     #with roi region
#        cv2.imshow("detect",detect)
#        print(detect)
        print("---------------------------VLP---------------------------------------------")
        #Increasing the contrast of image
        alpha = 1.5 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        detect = cv2.convertScaleAbs(detect, alpha=alpha, beta=beta)
#        cv2.imshow("before denoise",detect)
        detect = cv2.fastNlMeansDenoisingColored(detect, None, 10, 10, 7, 15) 
        
        scale_percent = 220 # percent of original size increase of it value will increases the size of the image.
        width = int(detect.shape[1] * scale_percent / 100)
        height = int(detect.shape[0] * scale_percent / 100)
        dim = (width, height)

        
        #resized image
        detect = cv2.resize(detect, dim, interpolation = cv2.INTER_AREA)
#        cv2.imshow("after denoise",detect)
#        cv2.waitKey(0)

        
        print("--------text here-------------starts")
        gray = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Final detection",gray)
#        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#        cv2.imshow("thres",thresh)
#        invert = cv2.bitwise_not(thresh)
#        cv2.imshow("invert",invert)
#        cv2.waitKey(0)
        text = pytesseract.image_to_string(gray)
        print("---------------------------------printing vehicle number plate--------------------") 
        print(text)
       
#        cv2.waitKey(0)
  
    cv2.imshow('video', image)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()