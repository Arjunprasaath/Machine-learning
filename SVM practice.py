 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Social_Network_Ads.csv')
x=data.iloc[:,2:4].values
y=data.iloc[:,4].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit_transform(xtrain)
ss.transform(xtest)

from sklearn.svm import SVC
svc=SVC(kernel='linear',random_state=0)
svc.fit(xtrain,ytrain)

y_pred=svc.predict(xtest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred)
 
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set =xtrain, ytrain
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svc.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()