#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:35:12 2018

@author: student
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def createRTable(contour,orientacja,cx,cy):
    
    RTable =  [[] for i in range(360)]
    
    for point in contour:
        pointGradient = orientacja[point[0][0]][point[0][1]]
        xx = cx-point[0][0]
        yy = cy-point[0][1]
        
        angle = math.floor(pointGradient*180/(math.pi))
        
        if angle < 0:
            angle=angle+360
        
        leng = math.sqrt(xx*xx+yy*yy)
        angle_point = np.arctan2(yy,xx)
        angle_point = math.floor(angle_point*180/(math.pi))
        RTable[angle].append((angle_point,leng))
        

I = cv2.imread('obrazy_hough/trybik.jpg')
I_g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I_g = 255*np.uint8(I_g < 160)
Im, contours, hierarchy = cv2.findContours(I_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

sobelx = cv2.Sobel(I_g,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(I_g,cv2.CV_64F,0,1,ksize=5)

gradient = np.sqrt(sobelx*sobelx + sobely*sobely)
gradient/=np.max(gradient)

orientacja = np.arctan2(sobelx,sobely)

M = cv2.moments(I_g,1)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

contour = contours[1]

createRTable(contour,orientacja,cx,cy)


#cv2.drawContours(I, contours, 1, (0,255,0), 1)
cv2.imshow("I",I)
