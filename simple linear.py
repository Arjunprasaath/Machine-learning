import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv('Salary_Data.csv')
print(data)
x= data.iloc[:,:-1].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/3)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)


plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,lr.predict(x_test),color='black')
plt.xlabel('experience')
plt.ylabel('salary')
plt.title('exp VS sal(test)')
plt.show()