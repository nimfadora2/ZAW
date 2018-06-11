import pickle
from sklearn import svm
import numpy as np


f = open('hog.pckl', 'rb')
HOG_data = pickle.load(f)
f.close()

# learning set
learn = HOG_data[0:int(0.6*len(HOG_data))]
data = learn[:,1:]
labels = learn[:,0]

# valid set
valid = HOG_data[int(0.6*len(HOG_data)):int(0.8*len(HOG_data))]
data_valid = valid[:,1:]
labels_valid = valid[:,0]

# test set
test = HOG_data[int(0.8*len(HOG_data)):len(HOG_data)]
data_test = test[:,1:]
labels_test = test[:,0]

clf = svm.SVC(kernel='linear', C = 4.0)
clf.fit(data,labels)
lp = clf.predict(data)
lpv = clf.predict(data_valid)
lpt = clf.predict(data_test)

T = np.logical_not(np.logical_xor(lp,labels))
TN = np.logical_and(np.logical_not(labels),np.logical_not(lp))
TP = np.logical_and(labels,lp)
FP = np.logical_and(np.logical_not(labels),lp)
FN = np.logical_and(np.logical_not(lp),labels)

Tv = np.logical_not(np.logical_xor(lpv,labels_valid))
TNv = np.logical_and(np.logical_not(labels_valid),np.logical_not(lpv))
TPv = np.logical_and(labels_valid,lpv)
FPv = np.logical_and(np.logical_not(labels_valid),lpv)
FNv = np.logical_and(np.logical_not(lpv),labels_valid)

Tt = np.logical_not(np.logical_xor(lpt,labels_test))
TNt = np.logical_and(np.logical_not(labels_test),np.logical_not(lpt))
TPt = np.logical_and(labels_test,lpt)
FPt = np.logical_and(np.logical_not(labels_test),lpt)
FNt = np.logical_and(np.logical_not(lpt),labels_test)

print(sum(Tt*1)/len(Tt))