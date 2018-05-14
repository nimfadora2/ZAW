#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:50:40 2018

@author: student
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('vid1_IR.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (tresh, G) = cv2.threshold(G,45,255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5))
    G = cv2.morphologyEx(G, cv2.MORPH_OPEN, kernel)
    G = cv2.morphologyEx(G, cv2.MORPH_CLOSE, kernel)
    
    out = cv2.connectedComponentsWithStats(G, 4, cv2.CV_32S)
    stats = out[2]
    
    points = []
    for i in range(out[0]):
        if not 4*stats[i,cv2.CC_STAT_HEIGHT] > stats[i,cv2.CC_STAT_WIDTH]:
            continue
        pt1 = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP])
        pt2 = (pt1[0]+stats[i, cv2.CC_STAT_WIDTH], pt1[1]+stats[i, cv2.CC_STAT_HEIGHT])
        points.append((pt1,pt2))
        
    good_points = []
    flags = [0]*len(points)
    
    for i in range(len(points)):
        up_Top = points[i][0][1]
        up_Bottom = points[i][1][1]
        up_Left = points[i][0][0]
        up_Right = points[i][1][0]
        for j in range(len(points)):
            down_Top = points[j][0][1]
            down_Bottom = points[j][1][1]
            down_Left = points[j][0][0]
            down_Right = points[j][1][0]
            if (up_Bottom - 10 < down_Top and (up_Left - 10 < down_Left) and (up_Right + 10 > down_Right)):
                good_points.append(((min(up_Left,down_Left),min(up_Top,down_Top)),(max(up_Right,down_Right),max(up_Bottom,down_Bottom))))
                flags[i]=1
                flags[j]=1
    
    for i in range(len(flags)):
        if flags[i]==0:
            good_points.append(points[i])
        
    
    for i in range(len(good_points)):
        punkt = good_points[i]
        cv2.rectangle(G,punkt[0],punkt[1],(255,0,0))
    
    cv2.imshow('IR',G)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
cap.release()