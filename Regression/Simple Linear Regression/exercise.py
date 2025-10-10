import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import Data
ds = pd.read_csv(r'Regression\Simple Linear Regression\salary_data.csv')

# Indepentend variables and dependent variable
X = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

# Trainning the Simple Linear Regression model on the Trainning set
regressor = LinearRegression() 
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualing the Training set results
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualing the Test set results
plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue') # The predicted salaries from the test set will be on the same regression line of the predicted salaries from the training set so we dont have to replace the value of X_train
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary of year 12
print(regressor.predict([[12]])) 

# Coefficients
print(regressor.coef_)
print(regressor.intercept_)

