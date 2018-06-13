# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:09:14 2018

@author: 15216
"""

import cv2  
  
cap = cv2.VideoCapture('S.mp4')  
i=0
while(cap.isOpened()):
   
    ret, frame = cap.read()
   
    
    img =frame[130:300, 530:700]
    # This parameter should be set in find_best_area.py
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray=cv2.blur(gray,(9,9))
    ret, thresh = cv2.threshold(gray,100, 255, cv2.THRESH_BINARY) 
      
    # Find Contour  
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
      
    # 需要搞一个list给cv2.drawContours()才行！！！！！
          
    cv2.drawContours(thresh, contours, -1, (255, 255, 255), thickness=-1)
    
    cv2.imshow("image",thresh)
    thresh=cv2.resize(thresh,(50,50))
  #  cv2.imwrite("mask.png", thresh) 
    cv2.imwrite("my_gestures/"+"1/"+str(i)+".jpg",thresh)
    i=i+1

    k = cv2.waitKey(20)  
   

cap.release()  
cv2.destroyAllWindows()




 