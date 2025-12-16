import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

# Import Data
dataset = pd.read_csv(r'10 - Model Selection & Boosting\Datasources\cancer_data.csv')

# Indepentend variables and dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Training the CatBoost model on the Training set
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)

# Predict the Test Results
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying K-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))