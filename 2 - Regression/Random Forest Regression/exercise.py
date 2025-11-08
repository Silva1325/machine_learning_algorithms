import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Import Data
ds = pd.read_csv(r'2 - Regression\Datasources\position_salaries_data.csv')

# Independent and dependent variables
X = ds.iloc[:, 1:-1].values
y = ds.iloc[:, -1].values 

# Train Random Forest Regression
regressor = RandomForestRegressor(n_estimators=10, random_state=0) # 10 Trees
regressor.fit(X, y)

# Predict on test data
regressor.predict(X)

# Predict a Single Value
print(regressor.predict([[6.5]]))

# Visualizing the Random Forest Regression results (Higher Resolution)
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()