import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

# Import Data
dataset = pd.read_csv(r'8 - Deep Learning\datasources\churn_modelling_data.csv')

# Indepentend variables and dependent variable
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# Encoding categorical data
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
#print(X)

# One Hot Encoding the Geography column
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)

# Building the ANN
ann = tf.keras.models.Sequential()
# Add input layer and the first hidden layer
# How do we know how many hidde layers do we want? Experiment.
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) 
# Add second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Add the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # softmax ( non binary classification)

# Compiling the ann
# crossentropy loss ( non binary classification)
ann.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

#ann.fit(X_train,y_train, batch_size = 32, epochs = 100)
ann.fit(X_train,y_train, batch_size = 32, epochs = 20)

prediction = ann.predict(sc.transform(np.array([[1,0,0,600,1,40,3,60000,2,1,1,50000]])) > 0.5)

print(prediction)

# Predict the Test Results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = accuracy_score(y_test, y_pred)
print(ac)