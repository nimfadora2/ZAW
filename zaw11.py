#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:39:36 2018

@author: student
"""

import cv2
import numpy as np
from numpy.fft import fftshift,fft2,ifft2,ifftshift
import matplotlib.pyplot as plt

### Pierwsza czesc ###
'''
wzor = cv2.imread('obrazy_Mellin/wzor.pgm')
wzor = cv2.cvtColor(wzor,cv2.COLOR_BGR2GRAY)
dom = cv2.imread('obrazy_Mellin/domek_r0.pgm')
dom = cv2.cvtColor(dom,cv2.COLOR_BGR2GRAY)

wzor_w, wzor_h = wzor.shape
dom_w, dom_h = dom.shape

nowy_wzor = np.zeros((128,128))

nowy_wzor[40:88,40:88]=wzor
wzor_przed = nowy_wzor
nowy_wzor = fftshift(nowy_wzor)


domFt = fftshift(fft2(dom))
wzorFt = fftshift(fft2(nowy_wzor))

ccor = np.conj(wzorFt)*domFt
modul = abs(ccor)

last = ifft2(ifftshift(ccor/modul))

y,x = np.unravel_index(np.argmax(last), last.shape)

dx = x - dom_w//2
dy = y - dom_h//2

macierz_tr = np.float32([[1,0,dx],[0,1,dy]])
obraz_przesuniety = cv2.warpAffine(wzor_przed,macierz_tr,
                                   (nowy_wzor.shape[1], nowy_wzor.shape[0]))

plt.imshow(obraz_przesuniety)
plt.plot([x],[y],'*',color='r')
plt.show()
'''

### Druga czesc ###
def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h,h))

def highpassFilter(size):
    rows = np.cos(np.pi*np.matrix([-0.5 + x/(size[0]-1) for x in range(size[0])]))
    cols = np.cos(np.pi*np.matrix([-0.5 + x/(size[1]-1) for x in range(size[1])]))
    X = np.outer(rows,cols)
    return (1.0 - X) * (2.0 - X)


dom = cv2.imread('obrazy_Mellin/domek_r30.pgm')
dom = cv2.cvtColor(dom,cv2.COLOR_BGR2GRAY)
wzor = cv2.imread('obrazy_Mellin/domek_r0_64.pgm')
wzor = cv2.cvtColor(wzor,cv2.COLOR_BGR2GRAY)

wzor_h = hanning2D(64)

nowy_wzor = np.zeros((128,128))
nowy_wzor[32:32+64,32:32+64]=wzor_h

domFt = fftshift(fft2(dom))
wzorFt = fftshift(fft2(nowy_wzor))

filtr = highpassFilter(nowy_wzor.shape)

dom_abs = abs(domFt)
wzor_abs = abs(wzorFt)

dom_filtr = filtr*dom_abs
wzor_filtr = filtr*wzor_abs

R = 64
M = 2*R/np.log(R)

dom_log = cv2.logPolar(dom_filtr,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS)
wzor_log = cv2.logPolar(wzor_filtr,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS)

