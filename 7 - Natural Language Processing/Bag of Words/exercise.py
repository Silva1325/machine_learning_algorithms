import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Import dataset
ds = pd.read_csv(r'7 - Natural Language Processing\datasources\restaurant_reviews_data.tsv', delimiter='\t', quoting=3) # Ignore quoting

# Cleaning the texts
nltk.download('stopwords') # The, a, they, them,...
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', ds.iloc[i]['Review'])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    # Stem words. Loved -> love ( Simplifying words )
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Bag of wrods model
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = ds.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train the naive bayes model 
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predict the Test Results
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = accuracy_score(y_test, y_pred)
print(ac)


