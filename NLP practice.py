
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Restaurant_Reviews.tsv', delimiter ='\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

corpus=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [PorterStemmer().stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x =cv.fit_transform(corpus).toarray()
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.30)
 

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=700,criterion='gini')
rfc.fit(xtrain,ytrain)

y_pred = rfc.predict(xtest) 

from sklearn.metrics import confusion_matrix
cm =  confusion_matrix(ytest,y_pred)
