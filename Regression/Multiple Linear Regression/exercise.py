import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Importing the dataset
ds = pd.read_csv(r'Regression\Multiple Linear Regression\50_startups.csv')

# Indepentend variables and dependent variable
X = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

# Trainning the Multiple Linear Regression model on top of trainning set
regressor = LinearRegression() 
regressor.fit(X_train,y_train)

# Predict the Test Results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Predict a new result with the model
# Example: R&D Spend=160000, Administration=130000, Marketing Spend=300000, State=California
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))