"""
Simple Pattern Recognition using a Gaussian model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv,det

def gauss(x,u,inv_s,det_s,k):
    tmp = x-u
    A = (1/np.sqrt((2*np.pi)**k * det_s))
    arg = -0.5*tmp.reshape(1,k).dot(inv_s.dot(tmp))
    return A * np.exp(arg)

class GaussModel:
    def __init__(self,mus,sigmas,dim):
        self.mus = mus
        self.sigmas = sigmas
        self.det_sigmas = [det(s) for s in sigmas]
        self.inv_sigmas = [inv(s) for s in sigmas]
        self.dim = dim
    def get_max_index(self,arr):
        max_value = max(arr)
        for i in range(self.dim):
            if arr[i] == max_value:
                return i
        return None
    def predict(self,x):
        pred = [[gauss(x0,u,inv_s,det_s,self.dim) for u,inv_s,det_s in zip(self.mus,self.inv_sigmas,self.det_sigmas)] for x0 in x]
        return np.array([self.get_max_index(arr) for arr in pred])

def get_model(x_train,y_train, num_classes):
    x_train += np.random.random(x_train.shape)
    _,dim = x_train.shape
    N = num_classes # Number of classes
    mu = np.zeros((N,dim)) # mean value for each class
    counter = np.zeros(N) # count the number of data of each class
    sigma = np.zeros((N,dim,dim)) # standard deviation for each class
    for u,c in zip(x_train,y_train):
        mu[c] += u
        counter[c] += 1
    mu /= counter.reshape(N,1)
    for u,c in zip(x_train,y_train):
        tmp = (u - mu[c])
        sigma[c] += tmp.reshape(dim,1).dot(tmp.reshape(1,dim))
    sigma /= counter.reshape(N,1,1)
    return GaussModel(mu,sigma,dim)

if __name__ == "__main__":
    samples = 50
    x1 = 2*np.random.random(samples) - 1
    y1 = 2*np.random.random(samples) - 1
    x_train = np.array([[i,j] for i,j in zip(x1,y1)])
    y_train = np.array([0 if j>0.5*i else 1 for i,j in x_train])
    model = get_model(x_train,y_train,2)
    pred = model.predict(x_train)
    
    # for i,j in zip(pred,y_train):
    #     print(i,j)
    
    # Plotting the prediction for several points in the region of [-1,1] x [-1,1]
    x1 = np.linspace(-1,1,50)
    y1 = np.linspace(1,-1,50)
    points = np.array([[i,j] for i in x1 for j in y1])
    pred = model.predict(points)
    plt.imshow(pred.reshape(50,50).T)
    plt.show()
            
