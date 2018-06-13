# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:11:36 2018

@author: 15216
"""

import cv2
cap = cv2.VideoCapture('S.mp4')  
i=0
   
while(cap.isOpened()): 
    print(i)
    ret, frame = cap.read()  
    cv2.imshow('image', frame) 
    
    crop_img =frame[130:300, 530:700]
    cv2.imshow("image", crop_img)
    k = cv2.waitKey(20)  
    i=i+1
    

cap.release()  
cv2.destroyAllWindows()