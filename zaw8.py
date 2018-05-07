# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import matplotlib.pyplot as plt
import math
import os
import numpy as np

def middle(c):

    M=cv2.moments(c)

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if i==0 and j==1:
                distance = (x[i]-x[j])**2+(y[i]-y[j])**2
            else:
                dist2 = (x[i]-x[j])**2+(y[i]-y[j])**2
                if dist2 > distance:
                    distance = dist2
    return math.sqrt(distance), cx,cy

def hausdorf(c_a):
    dist = []
    lista = os.listdir('imgs')
    for elem in lista:
        name = 'imgs/' + elem
        img_temp = cv2.imread(name)
        it_g = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        IT, contours,hierarchy = cv2.findContours(it_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        c = contours[1]
        x = c[:,0,0]
        y = c[:,0,1]
        distance,cx,cy = middle(c)
        
        x-=cx
        x=x/distance
        y-=cy
        y=y/distance
        
        dist.append(compare(c_a,c))
    return dist
        
        
def compare(c1,c2):
    x1 = c1[:,0,0]
    y1 = c1[:,0,1]

    x2 = c2[:,0,0]
    y2 = c2[:,0,1]
    
    point1 = np.Inf
    point2 = np.Inf
    
    for i in range(len(c1)):
        xx = x2-x1[i]
        yy = y2-y1[i]
        
        norma = [xel*xel+yel*yel for xel in xx for yel in yy]
        point1 = min(norma)
    
    
    for i in range(len(c2)):
        xx = x1-x2[i]
        yy = y1-y2[i]
        
        norma = [xel*xel+yel*yel for xel in xx for yel in yy]
        point2 = min(norma)
    
    return np.sqrt(max(point1,point2))
    

I=cv2.imread('imgs/c_astipalea.bmp')
I_g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
Im, contours,hierarchy = cv2.findContours(I_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

c = contours[1]
x = c[:,0,0]
y = c[:,0,1]

M=cv2.moments(c)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

distance,cx,cy = middle(c)

x-=cx
x=x/distance
y-=cy
y=y/distance

#compare(c,c)
hausdorf(c)

cv2.drawContours(I, contours, 1, (0,255,0), 3)
plt.imshow(I)