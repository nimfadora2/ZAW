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
	x = c[:, 0, 0]
	y = c[:, 0, 1]

	M = cv2.moments(c,1)

	cx = sum(x) / len(x)
	cy = sum(y) / len(y)

	x = np.float64(x) - cx
	y = np.float64(y) - cy

	distance = 0

	for i in range(len(x)):
		for j in range(i + 1, len(x)):
			dist2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
			if dist2 > distance:
				distance = dist2
	distance=math.sqrt(distance)
	x = x / distance
	y = y / distance

	return distance, cx,cy, x,y

def hausdorf(c_a):
	distance_a, cx_a, cy_a, x_a, y_a = middle(c_a)
	dist = []
	lista = os.listdir('plikiHausdorff/imgs')
	for elem in lista:
		name = 'plikiHausdorff/imgs/' + elem
		img_temp = cv2.imread(name)
		#print(name)
		it_g = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
		IT, contours, hierarchy = cv2.findContours(it_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		c = contours[0]
		distance, cx, cy, x,y = middle(c)
		dist.append(compare(x,x_a,y,y_a))
	return dist

def compare(x1,x2,y1,y2):
	point1 = 0
	point2 = 0

	for i in range(len(x1)):
		xx = x2 - x1[i]
		yy = y2 - y1[i]
		xx = xx**2
		yy = yy**2
		norma = xx+yy
		if point1 < min(norma):
			point1 = min(norma)

	for i in range(len(x2)):
		xx = x1 - x2[i]
		yy = y1 - y2[i]
		xx = xx*xx
		yy = yy*yy
		norma = xx+yy
		if point2 < min(norma):
			point2 = min(norma)

	return np.sqrt(max(point1, point2))

def compare_2(c1,c2):
	dist1,cx1,cy1,x1,y1 = middle(c1)
	dist2, cx2, cy2, x2, y2 = middle(c2)

	return compare(x1,x2,y1,y2)

def haudorf_all():
	A = cv2.imread('plikiHausdorff/Aegeansea.jpg')
	A_g = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
	out = np.uint8((A_g[:, :, 0] < 60) & (A_g[:, :, 1] > 30))

	Im, contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	contours = list(filter(lambda el: el.shape[0] > 20 and el.shape[0] < 2000, contours))

	lista = os.listdir('plikiHausdorff/imgs')
	for elem in lista:
		name = 'plikiHausdorff/imgs/' + elem
		print(name)
		I = cv2.imread(name)
		I_g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		Im, contours_image, hierarchy = cv2.findContours(I_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		c = contours_image[1]


		podobienstwo = np.Inf
		for i in range(len(contours)):
			kontur = contours[i]
			simil = compare_2(c, kontur)
			if simil < podobienstwo:
				podobienstwo = simil
				numer =i

		kontur = contours[numer]
		dist1, gx, gy, gx1, gy1 = middle(kontur)
		nazwa = elem.split('.')[0].split('_')[1]
		cv2.putText(A, nazwa, (int(gx), int(gy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

	plt.imshow(A)
	plt.show()


I = cv2.imread('plikiHausdorff/imgs/c_mykonos.bmp')
I_g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
Im, contours, hierarchy = cv2.findContours(I_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
c = contours[1]
distance, cx, cy,x,y = middle(c)

# compare(c,c)
#dist = hausdorf(c)

### 2.5 ##

'''
A = cv2.imread('plikiHausdorff/Aegeansea.jpg')
A_g = cv2.cvtColor(A,cv2.COLOR_BGR2HSV)
out = np.uint8((A_g[:,:,0] < 60) & (A_g[:,:,1] > 30))

Im, contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = list(filter(lambda el: el.shape[0]>20 and el.shape[0]<2000,contours))

podobienstwo=np.Inf
for i in range(len(contours)):
	kontur = contours[i]
	simil = compare_2(c,kontur)
	if simil < podobienstwo:
		podobienstwo=simil
		numer = i

print(numer,podobienstwo)

#numer = 73
kontur = contours[numer]
dist1,gx,gy,gx1,gy1 = middle(kontur)
cv2.putText(A, "Mykonos",(int(gx),int( gy)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255))
cv2.drawContours(A, contours, numer, (0,255,0), 10)
'''
haudorf_all()


