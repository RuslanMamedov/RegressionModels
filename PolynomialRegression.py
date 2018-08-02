# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 11:14:27 2018

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

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.2,random_state=0)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train) #applying linear regression to the polynomial subset

#Prediction
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

#Assessing the quality of fit
from sklearn.metrics import r2_score
print ('R-squared score for this model is ',r2_score(y_test,y_pred))



