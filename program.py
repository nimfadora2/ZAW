#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:43:08 2018

@author: student
"""
import cv2
import numpy as np

def rgb2gray(I):
    return 0.299*I[:,:,0] + 0.587*I[:,:,1] + 0.114*I[:,:,2]

i=1
I_prev = cv2.imread('office/input/in%06d.jpg' % i)
I_prev_g = cv2.cvtColor(I_prev,cv2.COLOR_BGR2GRAY)



for i in range(2,2051):
    
    I = cv2.imread('office/input/in%06d.jpg' % i)
    I_g = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    
    D = abs(1.0*I_g-1.0*I_prev_g)/255
    
    B=np.uint8(1*(D>0.04)*255)
    B = cv2.medianBlur(B,3)
    
    kernel = np.ones((3,3),np.uint8)
    B = cv2.erode(B,kernel,iterations = 1)
    
    kernel2 = np.ones((2,2),np.uint8)
    B2 = cv2.dilate(B,kernel2, iterations=2)
    
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(B2)
    
    #cv2.imshow("Labels",np.uint8(labels/stats.shape[0]*255))
    #cv2.imshow("I",B2)
    
    if (stats.shape[0] > 1):   # czy sa jakies obiekty
        pi, p =max(enumerate(stats[1:,4]), key=(lambda x: x[1]))
        pi = pi + 1
        # wyrysownie bbox
        cv2.rectangle(B2,(stats[pi,0],stats[pi,1]),(stats[pi,0]+stats[pi,2],stats[pi,1]+stats[pi,3]),(255,0,0),2)
        # wypisanie informacji
        cv2.putText(B2,"%f" % stats[pi,4],(stats[pi,0],stats[pi,1]),cv2.
                    FONT_HERSHEY_SIMPLEX,0.5,(255,0,0))
        cv2.putText(B2,"%d" %pi,(np.int(centroids[pi,0]),np.int(centroids[pi,1])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    cv2.imshow("I",B2)
    
    I_prev_g = I_g
    cv2.waitKey(10)
    
