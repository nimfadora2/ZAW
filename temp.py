# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2

# Wczytanie obrazka Mandrila
I=cv2.imread('mandril.jpg')
cv2.imshow("Mandril",I)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("m.png",I)

print(I.shape)
print(I.size)
print(I.dtype)

