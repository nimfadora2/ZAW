#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:42:55 2018

@author: Kinga Slowik
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator

img = cv2.imread("fontanna1.jpg")
img = np.float64(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

img2 = cv2.imread("fontanna2.jpg")
img2 = np.float64(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY))

#cv2.imshow("i",img)
#cv2.waitKey()

def wart_H(img,ksize):
    k=0.05
    imx = cv2.Sobel(img,cv2.CV_64F, 1,0,ksize=ksize)
    imy = cv2.Sobel(img,cv2.CV_64F, 0,1,ksize=ksize)
    
    Ixx= imx*imx
    Iyy = imy*imy
    Ixy = imx*imy
    
    Ixx = cv2.GaussianBlur(Ixx,(ksize,ksize),0)
    Iyy = cv2.GaussianBlur(Iyy,(ksize,ksize),0)
    Ixy = cv2.GaussianBlur(Ixy,(ksize,ksize),0)
    
    detM = Ixx*Iyy-Ixy*Ixy
    trM = Ixx+Iyy
    
    return detM-k*trM*trM

def max_local(img,thr=0.1):
    shapy = img.shape
    height=shapy[0]
    width=shapy[1]
    points=[]
    thr = img.max()*thr
    
    for i in range(1,height-1):
        for j in range(1,width-1):
            
            ROI = img[i-1:i+2,j-1:j+2]
            k=np.unravel_index(np.argmax(ROI, axis=None),ROI.shape)
            
            if k[0]==1 and k[1]==1:
                value = ROI[1,1]
                point = [i,j]
                if value > thr:
                    points.append(point)
    return points

def draw(img,points):
    plt.imshow(img,cmap='gray')
    for point in points:
        plt.plot(point[1],point[0],'*',color='r')
    
def opisy(img, points,ksize):
    areas=[]
    Y,X = img.shape
    good_points = list(filter(lambda (x,y): y>=ksize and y<Y-ksize and x>=ksize and x<X-ksize, points))
    for point in good_points:
        ROI = img[point[0]-ksize:point[0]+ksize+1,point[1]-ksize:point[1]+ksize+1]
        ROI = ROI.flatten()
        areas.append([point,ROI])
    return areas
        
def compare(points1,points2, N):
    miary=[]
    for i in range(len(points1)-1):
        point1 = points1[i]
        for j in range(len(points2)-1):
            point2 = points2[j]
            miary.append([point1[0],point2[0],sum([abs(x) for x in list(map(operator.sub,point1[1],point2[1]))])])
    miary = sorted(miary, key=lambda x: x[2])
    miary = miary[0:N]
    print(miary)
    return miary
   
I = wart_H(img,7)
I2 = wart_H(img2,7)
print("Image")

points = max_local(I,0.1)
points2 = max_local(I2,0.1)
print("Points")

good_points = opisy(img,points,5)
good_points2 = opisy(img2,points2,5)
print("Opisy")

N=10
out = compare(good_points,good_points2,N)

#out=[[[299, 626], [372, 36], 6783], [[170, 431], [372, 36], 6906], [[214, 433], [372, 36], 7062], [[376, 212], [446, 741], 7269], [[281, 508], [371, 63], 7352], [[472, 404], [460, 735], 7360], [[312, 503], [424, 19], 7539], [[319, 113], [446, 741], 7750], [[280, 53], [424, 19], 7762], [[299, 626], [374, 707], 7770]]

out1 = [x[0] for x in out]
out2 = [x[1] for x in out]

#draw(img,out1)
draw(img2,out2)



