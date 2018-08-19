# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:15:24 2018

@author: shubham
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn import metrics,svm
tr_data=np.zeros((10*5000,28))
tr_target=np.zeros((10*5000))
ts_data=np.zeros((10*400,28))
ts_target=np.zeros((10*400))
count1=0
count2=0

for i in range(10):
    for j in range(1,5000):
        st='mnist_png\\training'+'\\'+str(i)+'\\s'+'('+str(j)+')'+'.png'
        im=plt.imread(st)
        tr_data[count1,:]=np.mean(im,1)    
        count1=count1+1
        tr_target[count1]=i

for i in range(10):
    for j in range(1,400):
        st='mnist_png\\testing'+'\\'+str(i)+'\\s'+'('+str(j)+')'+'.png'
        im=plt.imread(st)
        ts_data[count2,:]=np.mean(im,1)    
        count2=count2+1
        ts_target[count2]=i
ker=['rbf','linear','poly']    
for i in range(len(ker)):       
    sv=svm.SVC(kernel=ker[i])
    sv.fit(tr_data,tr_target)
    out=sv.predict(ts_data)
    print("Recognition Accuracy for digits with ",ker[i],"=",metrics.accuracy_score(out,ts_target))
#______________result__________________
"""
Experiment for 5 classes.
Recognition Accuracy for digits with  rbf = 0.811
Recognition Accuracy for digits with  linear = 0.864
Recognition Accuracy for digits with  poly = 0.2725
>>> tr_data.shape
(10000, 28)
>>> tr_target.shape
(10000,)
>>> ts_data.shape
(2000, 28)
>>> ts_target.shape
(2000,)
"""
"""
Experiment for 10 classes.
Recognition Accuracy for digits with  rbf = 0.6895
Recognition Accuracy for digits with  linear = 0.71425
Recognition Accuracy for digits with  poly = 0.10775
>>> tr_data.shape
(50000, 28)
>>> tr_target.shape
(50000,)
>>> ts_data.shape
(4000, 28)
>>> ts_target.shape
(4000,)
>>>  
"""
