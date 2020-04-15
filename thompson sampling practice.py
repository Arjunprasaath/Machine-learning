
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
N=10000
d=10
no_of_rewards_1 = [0] *d
no_of_rewards_0 = [0] *d 
ads_selected = []
total_rewards = 0

for n in range(0,N):
    ad=0
    max_teta=0
    for i in range(0,d):
        teta = random.betavariate(no_of_rewards_1[i] +1, no_of_rewards_0[i] + 1)
        if teta > max_teta :
            max_teta = teta
            ad = i
    ads_selected.append(ad)
    reward = data.values[n,ad]
    if reward == 1 :
        no_of_rewards_1[ad] = no_of_rewards_1[ad] + 1
    else:
        no_of_rewards_0[ad] = no_of_rewards_0[ad] + 1
    total_rewards = total_rewards + reward
    
plt.hist(ads_selected)
plt.title('thompson sampling')
plt.xlabel('ads')
plt.ylabel('times clicked')
plt.show()

    