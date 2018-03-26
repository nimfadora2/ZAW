#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:39:57 2018

@author: student
"""

import cv2
import numpy as np
import matplotlib as plt


def draw_flow(img, u, v, step, thresh, color):
    y, x = img.shape
    mm=(u*u+v*v)**0.5
    m=mm>thresh
    for j in range(0,y,step):
        for i in range(0,x,step):
            if m[j][i]:
                cv2.line(img, (i,j), (i+np.int32(u[j][i
                         ]), j+np.int32(v[j][i])), color)
    
I = cv2.imread("I.jpg")
J = cv2.imread("J.jpg")

I_g = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
J_g = cv2.cvtColor(J,cv2.COLOR_BGR2GRAY)



cv2.imshow("J_g",J_g)
cv2.imshow("I_g",I_g)

diff = cv2.absdiff(I_g,J_g)



W2 = 1
dX = 1
dY = 1

u = np.zeros((240,360), np.float32)
v = np.zeros((240,360), np. float32)

for j in range(W2+dY,240-W2-dY):
    for i in range(W2+dX,320-W2-dX):
        IO = np.float32(I_g[j-W2:j+W2+1,i-W2:i+W2+1])
        
        dd = np.ones((2*dY+1,2*dX+1), np.float32)
        dd = np.inf - dd
        
        for n in range(-dY,dY+1):
            for m in range(-dX,dX+1):
                JO = np.float32(J_g[j-W2+n:j+W2+1+n,i-W2+m:i+W2+1+m])
                dd[m+dX,n+dY]=np.sum(np.square(JO - IO))
        
        ind = np.unravel_index(np.argmin(dd, axis=None),dd.shape)
        
        u[j,i]=(ind[0]-dX)/dX
        v[j,i]=(ind[1]-dY)/dY
        
draw_flow(diff,u,v,5,1,(255,0,0))
cv2.imshow("Diff",diff)