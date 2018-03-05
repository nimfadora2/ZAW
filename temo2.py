#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:01:35 2018

@author: student
"""

import matplotlib.pyplot as mtl
import matplotlib
import cv2
import scipy

#I = mtl.imread('mandril.jpg')
I=cv2.imread('mandril.jpg')

'''
mtl.figure(1)
mtl.imshow(I)
mtl.title('Mandril')
mtl.axis('off')
mtl.show()

mtl.imsave('mandril.png',I)

x=[100,150,200,250]
y=[50,100,150,200]

mtl.plot(x,y,markersize=10)

from matplotlib.patches import Rectangle

fig,ax=mtl.subplots(1)

rect=Rectangle((0.40,0.50),0.50,0.100,fill=False,ec='r')
ax.add_patch(rect)
mtl.show()
'''
'''
IG=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
IHSV=cv2.cvtColor(I,cv2.COLOR_BGR2HSV)

mtl.figure(1)
mtl.imshow(IG)
mtl.imsave('mandrilIG.png',IG)

mtl.figure(2)
mtl.imshow(IHSV)
mtl.imsave('mandrilIHSV.png',IHSV)

print(IHSV[:,:,0])
print(IHSV[:,:,1])
print(IHSV[:,:,2])
'''
def rgb2gray(I):
    return 0.299*I[:,:,0] + 0.587*I[:,:,1] + 0.114*I[:,:,2]
'''
mtl.figure(3)
mtl.gray()
mtl.imshow(rgb2gray(I))
'''
'''
I_HSV=matplotlib.colors.rgb_to_hsv(I)
mtl.figure(4)
mtl.imshow(I_HSV)
'''
'''
#mtl.figure(5)
height,width=I.shape[:2]
scale=1.75
Ix2=cv2.resize(I,(int(scale*height),int(scale*width)))
cv2.imshow("Big_Mandril",Ix2)

#I_2 = scipy.misc.imresize(I,0.5)
#cv2.imshow("Big_2",I_2)
'''

lena = cv2.imread("lena.png")
mand = cv2.imread("mandrilIG.png")



import numpy as np
#cv2.imshow("C",np.uint8(lena))
#cv2.imshow("Mand",np.uint8(mand))

cv2.imshow("C",(lena))
cv2.imshow("Mand",(mand))

cv2.imshow("Sum",(lena+mand))
cv2.imshow("Sub",(lena-mand))
cv2.imshow("Mul",(lena*mand))
cv2.imshow("Div",lena/mand*50)