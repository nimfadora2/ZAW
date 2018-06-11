#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:41:34 2018

@author: student
"""
import math                                     # do PI
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2

def hist(I_H,yS,xS,kernel_size):
    hist_q = np.zeros((256,1),float)
    for jj in range(0,kernel_size):
        for ii in range(0,kernel_size):
            pixel_H = I_H[yS+jj,xS+ii]
            hist_q[pixel_H] = hist_q[pixel_H] + pixel_H*G[jj,ii]
    return hist_q

# Generowanie Gaussa
kernel_size = 75                            # rozmiar rozkladu
sigma = 10                                  # odchylenie std
x = np.arange(0, kernel_size, 1,float)      # wektor poziomy
y = x[:,np.newaxis]                         # wektor pionowy
x0 = y0 = kernel_size // 2                  # wsp. srodka
G = 1/(2*math.pi*sigma**2)*np.exp(-0.5*((x-x0)**2 + (y-y0)**2) / sigma**2)

# Rysowanie
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, G, color='b')
plt.show()

# Pochodne
G_y = np.diff(G,1,0);
G_y = np.append(G_y,np.zeros((1,kernel_size),float),0)    # dodanie dodatkowego wiersza
G_y = -G_y
G_x = np.diff(G,1,1);
G_x = np.append(G_x,np.zeros((kernel_size,1),float),1)    # dodanie dodatkowej kolumny
G_x = -G_x

kernel_size = 45                            # rozmiar rozkladu

def track_init(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.rectangle(I, (x-kernel_size//2, y- kernel_size//2), (x +
                      kernel_size, y + kernel_size), (0, 255, 0), 2)
        mouseX,mouseY = x,y
        
# Wczytanie pierwszego obrazka
I = cv2.imread('track00100.png')
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking',track_init)

# Pobranie klawisza
while (1):
    cv2.imshow('Tracking',I)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:   # ESC
        break
    
    
xS = mouseX-kernel_size//2
yS = mouseY-kernel_size//2

I_HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

I_H = I_HSV[:,:,0]
hist_q = np.zeros((256,1),float)
for jj in range(0,kernel_size):
    for ii in range(0,kernel_size):
        pixel_H = I_H[yS+jj,xS+ii]
        hist_q[pixel_H] = hist_q[pixel_H] + pixel_H*G[jj,ii]
        
        
xC,yC = xS,yS
hist_p = hist_q
for i in range(101,201):
    I = cv2.imread('track%05d.png' % i)
    I_HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    H = I_HSV[:,:,0]
    for j in range(10):
        hist_q = np.zeros((256,1),float)
        for jj in range(0,kernel_size):
            for ii in range(0,kernel_size):
                pixel_H = I_H[yS+jj,xS+ii]
                hist_q[pixel_H] = hist_q[pixel_H] + pixel_H*G[jj,ii]
        rho = np.sqrt(hist_q*hist_p)
        dx_l = 0
        dx_m = 0
        dy_l = 0
        dy_m = 0
        for jj in range(0,kernel_size):
            for ii in range(0,kernel_size):
                dx_l = dx_l + ii*rho[H[yC+jj,xC+ii]]*G_x[jj,ii]
                dx_m = dx_m + ii*G_x[jj,ii]
                dy_l = dy_l + jj*rho[H[yC+jj,xC+ii]]*G_y[jj,ii]
                dy_m = dy_m + jj*G_y[jj,ii]
        dx = dx_l/dx_m
        dy = dy_l/dy_m
        # Obliczanie nowych wspolrzednych
        xC = np.int(np.floor(xC + dx))
        yC = np.int(np.floor(yC + dy))
    cv2.circle(I,(xC,yC),10,(255,0,0),thickness=10)
    cv2.imshow('Tracking',I)
    cv2.waitKey(100)
    hist_p = hist_q

