import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import Data
dataset = pd.read_csv(r'Data Processing\Datasources\people_purchases_data.csv')

# Indepentend variables and dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Take care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding categorical data
# 1 - Independent data
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 2 - Dependent data
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

# Feature Scaling
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_train[:,3:])

