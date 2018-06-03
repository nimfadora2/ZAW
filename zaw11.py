#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:39:36 2018
@author: student
"""

import cv2
import numpy as np
from numpy.fft import fftshift,fft2,ifft2,ifftshift
import math
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

wzor_h = hanning2D(64)*wzor

nowy_wzor = np.zeros((128,128))
wzor_NH = np.zeros((128,128))
nowy_wzor[32:32+64,32:32+64]=wzor_h
wzor_NH[32:32+64,32:32+64]=wzor

domFt = fftshift(fft2(dom))
wzorFt = fftshift(fft2(nowy_wzor))

filtr = highpassFilter(nowy_wzor.shape)

dom_abs = abs(domFt)
wzor_abs = abs(wzorFt)

dom_filtr = filtr*dom_abs
wzor_filtr = filtr*wzor_abs

R = 64
M = 2*R/np.log(R)

dom_log = cv2.logPolar(dom_filtr, (64,64), M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
wzor_log = cv2.logPolar(wzor_filtr, (64,64), M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

domFT2 = fftshift(fft2(dom_log))
wzorFT2 = fftshift(fft2(wzor_log))

ccor = np.conj(wzorFT2)*domFT2
modul = abs(ccor)
last = ifft2(ifftshift(ccor/modul))
wsp_kata,wsp_logr = np.unravel_index(np.argmax((last)), last.shape)

if wsp_logr > wzorFT2.shape[1] // 2:
    wykl = wzorFT2.shape[1] - wsp_logr
else:
    wykl = - wsp_logr
skala = np.exp(1/M)** wykl

A =  (wsp_kata * 360.0 ) / dom_log.shape[0]
kat1 = 360-A
kat2 = kat1-180

srodekTrans = [math.floor((dom.shape[0] + 1) / 2), math.floor((dom.shape[1] + 1 ) / 2)]
macierz_tr = cv2.getRotationMatrix2D((srodekTrans[0], srodekTrans[1]), kat1, skala)
macierz_tr2 = cv2.getRotationMatrix2D((srodekTrans[0], srodekTrans[1]), kat2, skala)

obraz_przesuniety1 = cv2.warpAffine(wzor_NH,macierz_tr,
                                   (wzor_NH.shape[1], wzor_NH.shape[0]))
ob_przFt = fft2(fftshift(obraz_przesuniety1))

obraz_przesuniety2 = cv2.warpAffine(wzor_NH,macierz_tr2,
                                   (wzor_NH.shape[1], wzor_NH.shape[0]))
ob_przFt2 = fft2(fftshift(obraz_przesuniety2))

ccor_Ft = np.conj(ob_przFt)*domFt
modul_Ft = abs(ccor_Ft)
last_Ft = ifft2(ifftshift(ccor_Ft/modul_Ft))
y,x = np.unravel_index(np.argmax(last_Ft), last_Ft.shape)

ccor_Ft2 = np.conj(ob_przFt2)*domFt
modul_Ft2 = abs(ccor_Ft2)
last_Ft2 = ifft2(ifftshift(ccor_Ft2/modul_Ft2))
y2,x2 = np.unravel_index(np.argmax(last_Ft2), last_Ft2.shape)

if last_Ft[y][x] > last_Ft2[x2][y2]:
    xk = x
    yk = y
    wzorzeck = obraz_przesuniety1
else:
    xk=x2
    yk = y2
    wzorzeck = obraz_przesuniety2
print(x,y,x2,y2)
dx = xk - wzorzeck.shape[0]//2
dy = yk - wzorzeck.shape[1]//2
macierz_translacji = np.float32([[1,0,dx],[0,1,dy]])
obraz_przesuniety = cv2.warpAffine(wzorzeck, macierz_translacji, (wzorzeck.shape[1], wzorzeck.shape[0]))

plt.figure(1)
plt.imshow(obraz_przesuniety)
plt.figure(3)
plt.imshow(wzor_NH)
plt.figure(2)
plt.imshow(dom)
plt.show()