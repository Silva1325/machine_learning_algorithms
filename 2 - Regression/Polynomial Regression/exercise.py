import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

# Import Data
ds = pd.read_csv(r'2 - Regression\Datasources\position_salaries_data.csv')

# Indepentend variables and dependent variable
X = ds.iloc[:,1:-1].values
y = ds.iloc[:,-1].values
y = y.reshape(len(y),1)

# Feature Scalling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Trainning Linear Regression model on the whole dataset
lin_reg = LinearRegression() 
lin_reg.fit(X,y)

# Trainning the Polynomial Regression model on the whole dataset
pol_reg = PolynomialFeatures(degree=2)
X_poly = pol_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualizing the Linear Regression results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg_2.predict(pol_reg.fit_transform(X)),color = 'blue')
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynomial Regression results ( Higher Resolution )
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg_2.predict(pol_reg.fit_transform(X_grid)),color = 'blue')
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

