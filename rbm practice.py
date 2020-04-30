# Boltzmann Machines

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

movies = pd.read_csv('ml-1m/movies.dat',sep = '::', header = None,engine = 'python',encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::', header = None,engine = 'python',encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = '::', header = None,engine = 'python',encoding = 'latin-1')

''' Preparing the training set and the test set by importing them and converting them into numpy array with data type 
as int.'''
training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t',header = None)
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t',header = None)
test_set = np.array(test_set, dtype = 'int')

# Getting the maximum number of users and movies

nb_users =int( max(max(training_set[:,0]),max(test_set[:,0])) )
nb_movies = int( max(max(training_set[:,1]),max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
'''creating a function which gives a list of lists in which each list belongs to a single user and 
contains the reviews/ratings given by the user to the movies watched where the index resembles the movie
ID'''

def convert_list(data):
    new_list =[]
    for user_id in range(1,nb_users+1):
        movie_id = data[:,1][data[:,0] == user_id]
        rating_given = data[:,2][data[:,0] == user_id]
        ratings = np.zeros(nb_movies)
        ratings[movie_id -1] = rating_given
        new_list.append(list(ratings))
    return new_list
training_set = convert_list(training_set)
test_set = convert_list(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set ==0] = -1
training_set[training_set ==1] = 0
training_set[training_set ==2] = 0
training_set[training_set >=3] = 1

test_set[test_set ==0] = -1
test_set[test_set ==1] = 0
test_set[test_set ==2] = 0
test_set[test_set >=3] = 1

# Creating the architecture of the Neural Network
'''creating a rbm class which consist of 3 functions (sample_h , sample_v , train) and
 a constructor '''
class RBM():
    def __init__(self, nv, nh): # nv,nh are the visible and hidden nodes respectively
        self.w = torch.randn(nh ,nv) # w is the weight
        self.a = torch.randn(1,nv) # a is the bias of visible node
        self.b = torch.randn(1,nh) # b is the bias of hidden node
    def sample_h(self,x): # x is the input that corresponds to probability of h given v
        wx = torch.mm(x,self.w.t())
        activation = wx + self.b.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        wy = torch.mm(y,self.w)
        activation = wy + self.a.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk): # v0 - input vector, vk - visible nodes obtained after k sampling, ph0 - probability of h given v0 and phk -is the probability of h after k sampling
        self.w +=( torch.mm(v0.t() , ph0) - torch.mm(vk.t(),phk)).t()
        self.a += torch.sum((v0 - vk),0)
        self.b += torch.sum((ph0 - phk),0)

nv = len(training_set[0]) # can also take nb_movies also
nh = 100
batch_size = 100
rbm = RBM(nv , nh)        
        
# Training the RBM

epoch = 10
for i in range(0,epoch ):
    counter =0.0
    loss = 0
    for user_id in range(0 ,nb_users - batch_size, batch_size):
        v0 =training_set[user_id :user_id+ batch_size] #target node
        vk = training_set[user_id: user_id+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(0,10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] #'''we take all the indexes that had -1 in vk and updating it to -1 since we dont want the system to predict for vales that have -1'''
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        loss += torch.mean(torch.abs(v0[v0>= 0] - vk[v0>= 0])) 
        counter += 1
        print('epoch = '+str(epoch)+ 'loss = ' +str(loss/counter))
# Testing the RBM
test_loss = 0
counter_2 = 0.
for user_id in range(nb_users):
    v = training_set[user_id: user_id +1] # we are using the training set inputs to activate the neurons to get the predictions fot the test set
    vt = test_set[user_id :user_id +1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        counter_2 += 1.
print('test loss: '+str(test_loss/counter_2))

