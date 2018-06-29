import cv2
import numpy as np
import random


# based on: https://github.com/yangshiyu89/VIBE/blob/master/vibe_test.py

random.seed()

I = cv2.imread('highway/input/in000001.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

height, width = I.shape

## init background ###

def initBackground(I,N):
	height, width = I.shape
	bufor = np.zeros((height,width,N))
	for j in range(1,width-1):
		for i in range(1,height-1):
			for k in range(N):
				x = random.randint(-1,1)
				y = random.randint(-1,1)
				while x == 0 and y == 0:
					x = random.randint(-1, 1)
					y = random.randint(-1, 1)
				bufor[i,j,k]=I[i+x, j+y]
	return bufor

def calculatePicture(I,bufor,R):
	height, width = I.shape
	N = bufor.shape[2]
	I_out = np.zeros((height,width))
	for j in range(1,width-1):
		for i in range(1,height-1):
			counter = 0
			dist = np.abs(bufor[i,j,:]-I[i,j])
			counter = sum((dist < R)*1)
			if counter >= 2:
				I_out[i,j]=0
				actual = random.randint(0, 15)
				if actual == 0:
					swap = random.randint(0, N - 1)
					x = random.randint(-1, 1)
					y = random.randint(-1, 1)
					while x == 0 and y == 0:
						x = random.randint(-1, 1)
						y = random.randint(-1, 1)
					bufor[i, j, swap] = I[i + x, j + y]
			else:
				I_out[i,j]=255
	return np.uint8(I_out),bufor

def VIBE():
	I = cv2.imread('highway/input/in000001.jpg')
	I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	bufor = initBackground(I,20)
	for i in range(2,500):
		I = cv2.imread('highway/input/in%06d.jpg' % i)
		I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		print("Picture started",i)
		I_out,bufor = calculatePicture(I,bufor,30)
		cv2.imshow("I",I_out)
		cv2.waitKey(1)



VIBE()

cv2.waitKey()