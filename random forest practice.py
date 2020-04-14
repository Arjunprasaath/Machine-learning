import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values
 
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=1000)
rfr.fit(x,y)

y_pred=rfr.predict([[6.5]])

x_grid=np.arange(min(x),max(x),step=0.01)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color='cyan')
plt.plot(x_grid,rfr.predict(x_grid),color='pink')
plt.title('random forest regressor')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
