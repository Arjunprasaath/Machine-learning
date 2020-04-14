import numpy as np
import pandas as pd

data = pd.read_csv('50_Startups.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,4].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer([('state',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x),dtype=np.int)

#to avoid dummy variables trap
x=x[:,1:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

#bulding backward elimination
x=np.append(arr= np.ones((50,1)).astype(int), values=x,axis=1)

import statsmodels.api as sm

x_opt = x[:,[0,1,2,3,4,5]]
ols=sm.OLS(endog=y,exog=x_opt).fit()
ols.summary()


x_opt = x[:,[0,3]]
ols=sm.OLS(endog=y,exog=x_opt).fit()