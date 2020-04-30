
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Credit_Card_Applications.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range =(0,1))
x = sc.fit_transform(x)

from minisom import MiniSom
som = MiniSom(x = 10, y = 10 , input_len = 15 , sigma = 1.0 , learning_rate = 0.5)
som.random_weights_init(x)
som.train_random( data = x , num_iteration = 100)

from pylab import plot , show , pcolor , colorbar , bone

bone()
pcolor(som.distance_map())
colorbar()
marker =['o','s']
colors =['r','g']
for i,j in enumerate(x):
    w = som.winner(j)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         marker[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mapping = som.win_map(x)
fraud = np.concatenate((mapping[(4,3)],mapping[(5,3)]),axis = 0)
fraud = sc.inverse_transform(fraud)

z = data.iloc[:,1:].values


is_fraud = np.zeros(len(data))
for i in range(len(data)):
    if data.iloc[i,0] in fraud:
        is_fraud[i] = 1

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
z = ss.fit_transform(z)   


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units = 2,input_dim = 15,activation = 'relu'))
model.add(Dense(units = 3,activation = 'relu'))
model.add(Dense(units = 1,activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(z,is_fraud, batch_size = 1, epochs = 3)

y_pred = model.predict(z)

y_pred = np.concatenate((data.iloc[:,0:1].values,y_pred),axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()]

