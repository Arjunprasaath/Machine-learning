import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y.reshape(-1,1))

from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(x,y)

y_predict1=sc_y.inverse_transform(svr.predict(sc_x.transform([[6.5]])))

plt.scatter(x,y,color='orange')
plt.plot(x,svr.predict(x),color='gray')
plt.title('support vector regression')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()