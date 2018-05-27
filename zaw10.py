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


def createRTable(contour, orientacja, cx, cy):
	RTable = [[] for i in range(360)]
	angles = []
	lengs =[]

	for point in contour:
		pointGradient = orientacja[point[0][0]][point[0][1]]
		xx = cx - point[0][0]
		yy = cy - point[0][1]

		angle = int(math.floor(pointGradient * 180 / (math.pi)))

		if angle < 0:
			angle = angle + 360

		leng = math.sqrt(xx * xx + yy * yy)
		angle_point = np.arctan2(yy, xx)
		angle_point = math.floor(angle_point * 180 / (math.pi))
		RTable[angle].append((angle_point, leng))
		angles.append(angle_point)
		lengs.append(leng)

	return RTable, angles,lengs


I = cv2.imread('obrazy_hough/trybik.jpg')
I_g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I_g = 255 * np.uint8(I_g < 160)
Im, contours, hierarchy = cv2.findContours(I_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

sobelx = cv2.Sobel(I_g, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(I_g, cv2.CV_64F, 0, 1, ksize=5)

gradient = np.sqrt(sobelx * sobelx + sobely * sobely)
gradient /= np.max(gradient)

orientacja = np.arctan2(sobelx, sobely)

M = cv2.moments(I_g, 1)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

contour = contours[1]

RTable_I, angles, lengs = createRTable(contour, orientacja, cx, cy)

J = cv2.imread('obrazy_hough/trybiki2.jpg')
J_g = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
J_g = 255 * np.uint8(J_g < 160)
Im2, contours2, hierarchy2 = cv2.findContours(J_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

sobelx_j = cv2.Sobel(J_g, cv2.CV_64F, 1, 0, ksize=5)
sobely_j = cv2.Sobel(J_g, cv2.CV_64F, 0, 1, ksize=5)

gradient_j = np.sqrt(sobelx_j * sobelx_j + sobely_j * sobely_j)
gradient_j /= np.max(gradient_j)

orientacja_j = np.arctan2(sobelx_j, sobely_j)


### Część pierwsza - 2 wymiary ###
width,height = gradient_j.shape
acc =np.zeros((550,550))
'''
for x in range(width):
	for y in range(height):
		if gradient_j[x][y] > 0.5:
			xx = x + lengs*np.cos(np.deg2rad(angles))
			xx = np.int_(xx)
			yy = y + lengs*np.sin(np.deg2rad(angles))
			yy = np.int_(yy)
			for i in range(len(xx)):
				acc[xx[i]][yy[i]]+=1
			#for row in RTable_I:
				#for point in row:
					#xx = int(x + point[1]*np.cos(np.deg2rad(point[0])))
					#yy = int(y + point[1]*np.sin(np.deg2rad(point[0])))
					#acc[xx][yy]+=1
print(np.argmax(acc), np.unravel_index(np.argmax(acc),acc.shape))
# 315,367
# 143,198
'''

### Czesc druga - trzy wymiary ###
'''
acc_new = acc.shape + (36,)
acc_new = np.zeros(acc_new)
for x in range(width):
	print(x)
	for y in range(height):
		for angle in range(0,360,10):
			if gradient_j[x][y] > 0.5:
				ang = np.subtract(angles, angle)
				xx = x + lengs * np.cos(np.deg2rad(ang))
				xx = np.int_(xx)
				yy = y + lengs * np.sin(np.deg2rad(ang))
				yy = np.int_(yy)
				for i in range(len(xx)):
					acc_new[xx[i]][yy[i]][int(angle/10)] += 1

				#for row in RTable_I:
					#for point in row:
						#xx = int(x + point[1]*np.cos(point[0]-angle))
						#yy = int(y + point[1]*np.sin(point[0]-angle))
						#acc_new[xx][yy][int(angle/10)]+=1

maxy =[]
delta = 30
for i in range(5):
	maxy.append([np.argmax(acc_new), np.unravel_index(np.argmax(acc_new),acc_new.shape)])
	x,y,z = np.unravel_index(np.argmax(acc_new),acc_new.shape)
	print(x,y,z)
	acc_new[x-delta:x+delta,y-delta:y+delta,:] = 0
'''

# cv2.drawContours(I, contours, 1, (0,255,0), 1)
plt.imshow(I_g)
plt.show()
plt.imshow(J_g)
plt.plot([367],[315],'*',color='r')

plt.plot([321],[181],'*',color='r')
plt.plot([198],[142],'*',color='r')
plt.plot([102],[310],'*',color='r')
plt.plot([367],[315],'*',color='r')
plt.plot([229],[304],'*',color='r')

#cv2.imshow("I2",I)
#cv2.waitKey()
plt.show()