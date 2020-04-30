
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('Credit_Card_Applications.csv')

x = data.iloc[:,:-1].values

y = data.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))

x = sc.fit_transform(x) 

from minisom import MiniSom

som = MiniSom(x =20, y =20, input_len = 15,sigma = 0.1,learning_rate = 0.5)
som.random_weights_init(data = x)
som.train_random(data = x, num_iteration = 200)


from pylab import pcolor, bone, show, plot, colorbar
bone()

pcolor(som.distance_map().T)

colorbar()

marker = ['o','s']
colors = ['r','g']

for i ,g in enumerate(x):
    w = som.winner(g)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         marker[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'none',
         markeredgewidth = 2,
         markersize = 10)
show()

mapping = som.win_map(x)
fruads = np.concatenate((mapping[(18,4)], mapping[(10,15)] ),axis = 0)
fruads = sc.inverse_transform(fruads)
    
    
    
    
    