# Linear Regression vs. Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')
X = dataset.iloc[:, 5:6].values #picked square footage as an independent variable
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train) #creating polynomial subset for the independent variable
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train) #applying linear regression to the polynomial subset

# Visualising the Linear Regression results (for the sake of comparison)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lin_reg.predict(X_train), color = 'blue')
plt.title('Home Prices (Linear Regression)')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) #making it a matrix (number of raws and columns in parenthesis)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Home Prices (Polynomial Regression)')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.show()

# Predicting a new result with Linear Regression
print (lin_reg.predict(y_test))

# Predicting a new result with Polynomial Regression
print (lin_reg_2.predict(poly_reg.fit_transform(y_test)))