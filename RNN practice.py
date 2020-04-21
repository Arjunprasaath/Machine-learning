

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv('Google_Stock_Price_Train.csv') 

training_set = data_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler( feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

xtrain = []
ytrain = []

for i in range(60,1258):
    xtrain.append(training_set_scaled[i-60:i,0])
    ytrain.append(training_set_scaled[i,0])

xtrain, ytrain = np.array(xtrain) , np.array(ytrain)

xtrain = np.reshape(xtrain, (xtrain.shape[0],xtrain.shape[1],1))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout



model = Sequential()

model.add(LSTM(units = 50, input_shape = (xtrain.shape[1],1),return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50,return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50,return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])

model.fit(xtrain ,ytrain, batch_size = 32, epochs = 100)

#making prediction

data_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_result = data_test.iloc[:,1:2].values

total_data = pd.concat((data_train['Open'] , data_test['Open']),axis = 0)

inputs = total_data[len(total_data) - len(data_test)-60:].values
inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

xtest=[]

for i in range(60,80):
    xtest.append(inputs[i-60:i,0])
xtest = np.array(xtest)

xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))

prediction = model.predict(xtest)
prediction = sc.inverse_transform(prediction)


plt.plot(real_result,color = 'red',label='google stock price')
plt.plot(prediction,color = 'blue',label = 'predicted result')
plt.title('google stock price prediciton')
plt.xlabe('date')
plt.ylabel('price')
plt.legend()
plt.show()