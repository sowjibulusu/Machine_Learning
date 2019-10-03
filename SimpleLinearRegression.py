#Simple Linear Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pa

#Importing the Dataset
dataset = pa.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the dataset into Training Set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

'''#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
##''For Training Set we need to fit and Transform where as for test set we
##only need to transform, because it is already fitted to training set
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

#Fitting Simple Linar Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test Set Results
#y_pred = Vector of Prediction of dependent Variables
y_pred = regressor.predict(x_test)

#Visualising the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary v/s Experience Training Set")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set results
plt.scatter(x_test, y_test, color = 'green')
plt.plot(x_test, regressor.predict(x_test), color = 'orange')
plt.title("Salary v/s Experience Training Set")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()




