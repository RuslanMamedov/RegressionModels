# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 06:52:47 2018

@author: Mamedov
"""

#Importing the libraries
import pandas as pd
import numpy as np

#Importing the dataset
dataset=pd.read_csv('kc_house_data.csv')
X=dataset.drop(['price','date'], axis=1).values
y=dataset.iloc[:,2].values
y = y.reshape(-1,1)

#Applying backward elimination to optimize the dataset (dropping columns with low correlation to y)
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((21613,1)).astype(int), values=X, axis=1)  #to append 1-s as the first column

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
SL = 0.05
X= backwardElimination(X, SL)
X=np.delete(X, 0, axis=1)
#'floors' and coefficient [0] columns got dropped

#Splitting the dataset into a train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state = 0)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

#Making prediction for the test set
y_pred = regressor.predict(X_test)

#Assessing the quality of fit
from sklearn.metrics import r2_score
print ('R-squared score for this model is ',r2_score(y_test,y_pred))
 
