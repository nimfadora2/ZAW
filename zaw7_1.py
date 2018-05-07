import cv2
import numpy as np
from matplotlib import pyplot as plt

L = cv2.imread('calibration_stereo/aloes/aloeL.jpg')
L = cv2.cvtColor(L,cv2.COLOR_BGR2GRAY)
R = cv2.imread('calibration_stereo/aloes/aloeR.jpg')
R = cv2.cvtColor(R,cv2.COLOR_BGR2GRAY)

'''
### BM ###

stereo = cv2.StereoBM_create(numDisparities=80,blockSize=15)
disparity = stereo.compute(L,R)

plt.imshow(disparity,'gray')
plt.show()
'''
'''
### SGM ###

stereo = cv2.StereoSGBM_create(numDisparities=96,blockSize=21)
disparity = stereo.compute(L,R)

plt.imshow(disparity,'gray')
plt.show()
'''
def Census(n, L, R, d = 10):
	width, height = L.shape
	#print(width,height)
	for i in range(n,width-n):
		for j in range (n,height-n):
			ROI = L[i-n:i+n+1,j-n:j+n+1]
			ROI_B = 1*(ROI > ROI[n,n])
			if height - j- n - d > 0:
				iter = d
			else:
				iter = (height - n - j)
				#print(iter)
			diff = []
			for dd in range(iter):
				#print(dd, i, j)
				ROI_R = R[i-n:i+n+1,j-n+dd:j+n+1+dd]
				ROI_R_B = 1*(ROI_R > ROI_R[n,n])
				differences = np.logical_xor(ROI_B,ROI_R_B)
				#print(sum(sum(differences)))
				diff.append(sum(sum(differences)))
			min(diff)


Census(2,L,R)