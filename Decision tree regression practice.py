import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(criterion='mse')
dtr.fit(x,y)

y_pred=dtr.predict([[6.5]])

x_grid=np.arange(min(x),max(y),step=0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='blue')
plt.plot(x_grid,dtr.predict(x_grid),color='orange')
plt.title('decision tree regression')
plt.xlabel('levels')
plt.ylabel('salary')
plt.show()