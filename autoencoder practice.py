# Importing the libraries
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importing the dataset

movies = pd.read_csv('ml-1m/movies.dat', sep = '::' , header= None, encoding = 'latin-1',engine = 'python')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::' , header= None, encoding = 'latin-1',engine = 'python')
users = pd.read_csv('ml-1m/users.dat', sep = '::' , header= None, encoding = 'latin-1',engine = 'python')

# Preparing the training set and the test set

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t', header = None)
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t', header = None)
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies

nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = nb_users = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns

def convert_list(data):
    new_data = []
    for user_id in range(1,nb_users + 1):
        movies_rated = data[:,1][data[:,0]==user_id]
        ratings_got = data[:,2][data[:,0] ==user_id]
        
        ratings = np.zeros(nb_movies)
        ratings[movies_rated-1] = ratings_got
        new_data.append(list(ratings))
    return new_data

training_set = convert_list(training_set)
test_set = convert_list(test_set)

# Converting the data into Torch tensors

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network

class SAE(nn.Module):
    def __init__(self,):
        super().__init__()
        self.fc1 = nn.Linear(nb_movies, 40)
        self.fc2 = nn.Linear(40 , 10)
        self.fc3 = nn.Linear(10 , 40)
        self.fc4 = nn.Linear(40 , nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
        

# Training the SAE

nb_epochs = 300
for epochs in range(1,nb_epochs+1):
    train_loss =0
    counter = 0.
    for user_id in range(0,nb_users):
        input = Variable(training_set[user_id]).unsqueeze(0)
        '''what the above line of code does is it adds another dimension to the training set which 
        is accepted format'''
        target = input.clone()
        if (torch.sum(target.data >0)) > 0:
            output = sae(input)
            target.require_grad = False
            output[target ==0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data >0) +1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            counter += 1.
            optimizer.step()
    print('epochs ='+str(epochs)+' loss = '+str(train_loss/counter))
            
# Testing the SAE

test_loss =0
counter = 0.
for user_id in range(0,nb_users):
    input = Variable(training_set[user_id]).unsqueeze(0)  
    target = Variable(test_set[user_id])
    if (torch.sum(target.data >0)) > 0:
        output = sae(input)
        target.require_grad = False
        output[(target ==0).unsqueeze(0)] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data >0) +1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        counter += 1.
print(' loss = '+str(test_loss/counter))
            
