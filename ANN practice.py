
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('Churn_Modelling.csv')

x = data.iloc[:,3:-1].values

y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])

le2 = LabelEncoder()
x[:,2] = le2.fit_transform(x[:,2]) 

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x),dtype = np.int)

x = x[:,1:]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20) 

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
xtrain = ss.fit_transform(xtrain)
xtest = ss.transform(xtest)

model = Sequential()
model.add(Dense(input_dim = 11,output_dim = 36, activation= 'relu'))
model.add(Dense(output_dim = 36, activation= 'relu'))
model.add(Dense(output_dim =1 , activation= 'sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
          
model.fit(xtrain,ytrain,batch_size=10,epochs=100)

y_pred = model.predict(xtest) 
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,y_pred)
