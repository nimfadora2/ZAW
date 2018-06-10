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

# test set
test = HOG_data[int(0.8*len(HOG_data)):len(HOG_data)]

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(data,labels)
lp = clf.predict(data)

T = np.logical_not(np.logical_xor(lp,labels))
TN = np.logical_and(np.logical_not(labels),np.logical_not(lp))
TP = np.logical_and(labels,lp)
FP = np.logical_and(np.logical_not(labels),lp)
FN = np.logical_and(np.logical_not(lp),labels)



print(np.logical_not(np.logical_xor(lp,labels)))