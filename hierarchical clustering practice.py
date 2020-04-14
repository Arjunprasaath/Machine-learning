
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Mall_Customers.csv')
x=data.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('euclidean distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='careful')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='green',label='standard')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='cyan',label='targets')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='blue',label='careless')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='yellow',label='selfcontrolled')
plt.title('clusters of customers')
plt.xlabel('salary in dollers')
plt.ylabel('ranking(1-100)')
plt.legend()
plt.show()