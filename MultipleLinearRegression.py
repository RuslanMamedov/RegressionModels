#Importing the libraries
import pandas as pd
import numpy as np

#Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')
X = dataset.drop(['price', 'date'], axis=1).values 
y = dataset.iloc[:,2].values
y = y.reshape(-1,1)

#Backward elimination
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
X_modeled= backwardElimination(X, SL)
X_modeled=np.delete(X_modeled, 0, axis=1)
#floors' and coefficient [0] columns got dropped


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split (X_modeled,y,test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Making a prediction
y_pred = regressor.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print ('The mean accuracy for this model is ', accuracies.mean())
print ('The standard deviation is', accuracies.std())

#Assessing the quality of fit
from sklearn.metrics import r2_score
print ('R-squared score for this model is ',r2_score(y_test,y_pred))




