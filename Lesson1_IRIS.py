# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:58:11 2019

@author: okramer
"""
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import ShuffleSplit
from Utils.plot_learning_curve import plot_learning_curve
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np






raw_data = load_iris()

X = pd.DataFrame(data=raw_data["data"],columns=raw_data["feature_names"])
Y = pd.DataFrame(data=raw_data["target"],columns=["Classifikation"])
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)



c_forest = RFC(n_estimators=500,n_jobs=6,verbose=1)#
plot_learning_curve(c_forest, "learncurve tree", X, Y, ylim=(0.7, 1.01), cv=cv)
#c_forest.fit(X,Y)
#
#
#importances = c_forest.feature_importances_
#std = np.std([treee.feature_importances_ for treee in c_forest.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]
#print("Feature ranking:")
#
#for f in range(X.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]), indices)
#plt.xlim([-1, X.shape[1]])
#plt.show()