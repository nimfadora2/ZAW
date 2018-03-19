#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:38:31 2018

@author: student
"""
import cv2
import numpy as np
import queue

file = open('office/temporalROI.txt','r')
start,stop = [int(x) for x in file.readline().strip().split()]
file.close()

iStep = 3

YY = 240
XX = 360
N = 60
BUF = np.zeros((YY,XX,N),np.uint8)

P = []
R = []
F1 = []

iN = 0
Buf_full  = False

'''
    # bufor mediana i średnia
for i in range(start,stop,iStep):
    
    I=cv2.imread('office/input/in%06d.jpg' % i)
    I_g=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    
    M = cv2.imread('office/groundtruth/gt%06d.png' % i)

    BUF[:,:,iN]=I_g
    
    if iN == N-1:
        Buf_full = True
        iN = 0
    else:
        iN+=1
        

    if Buf_full == True:
        back = np.mean(BUF, axis=2)
        median = np.median(BUF, axis=2)
        
        back2 = abs(I_g - back)
        median2 = abs(I_g - median)
        
        
        back8 = np.uint8(255*(back2>30))
        median8 = np.uint8(255*(median2>30))
        
        cv2.imshow("Back", back8)
        cv2.imshow("Median", median8)
        



    cv2.imshow("I",I_g)
    cv2.imshow("M",M)
'''

# aproksymacja średnia i mediana
for i in range(start,stop,iStep):
    
    I=cv2.imread('office/input/in%06d.jpg' % i)
    I_g=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    
    M = cv2.imread('office/groundtruth/gt%06d.png' % i)

    alfa = 0.05
     
    if i == start:
        BG_mean = I_g
        BG_median = I_g
    else:
        BG_mean = alfa*I_g+(1-alfa)*BG_mean
        BG_median = BG_median+np.sign(I_g-BG_median)

    back = abs(I_g - BG_mean)
    median = abs(I_g - BG_median)
    
    back8 = np.uint8(255*(back>30))
    median8 = np.uint8(255*(median>30))
    
    cv2.imshow("Back", back8)
    cv2.imshow("Median", median8)

    cv2.imshow("I",I_g)
    cv2.imshow("M",M)

    cv2.waitKey(2)
    
