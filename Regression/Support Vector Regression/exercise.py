import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Import Data
ds = pd.read_csv(r'Datasources\position_salaries_data.csv')

# Indepentend variables and dependent variable
X = ds.iloc[:,1:-1].values
y = ds.iloc[:,-1].values
y = y.reshape(len(y),1)

# Feature Scalling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Trainning the SVR model on the whole dataset
regressor = SVR(kernel= 'rbf')
regressor.fit(X,y.ravel())

# Predict the new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

# Visualizing the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), 
         sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), 
         color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the SVR results ( For high resolution and smoother curve )
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, 
         sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), 
         color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
