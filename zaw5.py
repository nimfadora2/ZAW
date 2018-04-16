import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("fontanna1.jpg")
img = np.float64(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

def rozmycie(img,ilosc,fi,k):
	result = []
	img_temp = cv2.GaussianBlur(img,(0,0),fi)
	result.append(img-img_temp)
	for i in range(ilosc-1):
		img_prev = cv2.GaussianBlur(img,(0,0),fi)
		fi = fi*k
		img_next = cv2.GaussianBlur(img,(0,0),fi)
		result.append(img_prev-img_next)
	return result


def maxima(result, thr):
	point=[]
	print(len(result))
	for i in range(1,len(result)-1):
		print(i)
		img_prev = result[i-1]
		img_next = result[i+1]
		img = result[i]
		width,height = img.shape
		for x in range(1,width-1):
			for y in range(1,height-1):
				ROI_prev = img_prev[x-1:x+2,y-1:y+2]
				min_ROI_prev = ROI_prev.max()
				ROI_next = img_next[x - 1:x + 2, y - 1:y + 2]
				min_ROI_next = ROI_next.max()
				ROI = img[x - 1:x + 2, y - 1:y + 2]
				min_ROI = ROI.max()
				if min_ROI > min_ROI_next and min_ROI > min_ROI_prev and min_ROI>thr and min_ROI==img[x,y]:
					point.append((x,y,i))
	return point

result = rozmycie(img, 5, 1.6, 1.26)
points = maxima(result,2)
'''
image = result[1]
good = list(filter(lambda x:x[2]==1,points))
plt.imshow(abs(image), cmap='gray')
plt.plot([x[1] for x in good], [x[0] for x in good], '*')
plt.show()
'''
image = result[2]
good = list(filter(lambda x:x[2]==2,points))
plt.imshow(abs(image), cmap='gray')
plt.plot([x[1] for x in good], [x[0] for x in good], '*')
plt.show()
'''
image = result[3]
good = list(filter(lambda x:x[2]==3,points))
plt.imshow(abs(image), cmap='gray')
plt.plot([x[1] for x in good], [x[0] for x in good], '*')
plt.show()
'''