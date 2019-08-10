# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:03:36 2019

@author: okramer
"""
import sys,os

import pandas as pd
import numpy as np
import operator

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from Utils.plot_learning_curve import plot_learning_curve as plc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score




############################ Daten vorbereitungen ##########################################
data_path = r"D:\Data\titanic-machine-learning-from-disaster"

TrainData = pd.read_csv(os.path.join(data_path,"train.csv"))
TestData = pd.read_csv(os.path.join(data_path,"test.csv"))
erg = pd.read_csv(os.path.join(data_path,"gender_submission.csv"))

_Train_columns = list(TrainData.columns)
_Test_columns = list(TestData.columns)


TrainData = TrainData.replace(to_replace="male",value=1)
TrainData = TrainData.replace(to_replace="female",value=0)
Train_Target = TrainData["Survived"]
TrainData = TrainData.loc[:,["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare"]]
TrainData = TrainData.apply(lambda x: x.fillna(x.mean()),axis=0)



TestData = TestData.replace(to_replace="male",value=1)
TestData = TestData.replace(to_replace="female",value=0)
TestData = TestData.loc[:,["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare"]]
TestData = TestData.apply(lambda x: x.fillna(x.mean()),axis=0)


################################### pre sampling ++++++++++++++++++++++++++++++++++++++++++
##################### model bau

ann = MLP(hidden_layer_sizes=(10,4),activation='relu',solver='sgd',alpha = 0.0001,
         learning_rate = 'constant',learning_rate_init = 0.00001,max_iter =10000,
         tol = 0.00000000005)

cv = SSS(n_splits=20,test_size=0.3,random_state=42)
selector=SelectKBest(chi2, k=5)
TrainData_new = selector.fit_transform(TrainData, Train_Target)
selector.get_support(indices=True)
TrainData_new = TrainData_new
Train_Target = Train_Target.values


networks = {}
f1_scores = {}

i=0
for train_index, test_index in cv.split(TrainData_new, Train_Target):
    
    X_train, X_test = TrainData_new[train_index], TrainData_new[test_index]
    y_train, y_test = Train_Target[train_index], Train_Target[test_index]
    
    ann.fit(X_train,y_train)
    networks[i]=ann
    f1_scores[i]=f1_score(y_test,ann.predict(X_test))
    i=i+1
    
    
for i in f1_scores.keys():
    print ("{k} : {score}".format(k=i,score=f1_scores[i]))
    
best_ann = max(f1_scores.items(), key=operator.itemgetter(1))[0]

ANN = networks[best_ann]

################################### vorbereitung erg data +++++++++++++

shape_test = erg.shape

surviver = {}
for i in range(erg.shape[0]):
    surviver[erg.iloc[i,0]] = erg.iloc[i,1]


s_testdata = selector.transform(TestData)

test_data = {}
for i in range(TestData.shape[0]):
    test_data[TestData.iloc[i,0]] = s_testdata[i,:]
    
    
true = []
pred = []    
for k in test_data.keys():
    true.append(surviver[k])
    pred.append(test_data[k])
    
true =  np.asarray(true)
pred =  np.asarray(true)
print(f1_score(true, pred))




