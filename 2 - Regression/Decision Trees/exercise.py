import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Import Data
ds = pd.read_csv(r'2 - Regression\Datasources\position_salaries_data.csv')

# Independent and dependent variables
X = ds.iloc[:, 1:-1].values
y = ds.iloc[:, -1].values 

# Train Decision Tree
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predict on test data
y_pred = regressor.predict(X)

# Predict a Single Value
print(regressor.predict([[6.5]]))

# Visualizing the Decission Tree Regression results ( Higher Resolution )
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title("Truth or bluff(Decsision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
