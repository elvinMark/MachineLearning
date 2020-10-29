"""
Simple Fully Connected Neural Network
"""

"""First we need to import some useful libraries: numpy ans matplotlib

- numpy will allow us to work with matrices that we are going to use a lot in
  the code

- matplotlib will allow us to draw some graphs like the how the loss functions
  is evolving during the training

"""
import numpy as np
import matplotlib.pyplot as plt

"""
Activation Functions

We will define here some activation (the more famous ones) functions.

- Sigmoid
- Hyperbolic tangent
- Rectified linear unit (ReLU)
- Leaky Rectified Linear Unit (Leaky ReLU)
- Linear 
"""
def sigmoid(x,diff=False):
    if diff:
        return x*(1-x)
    return 1/(1 + np.exp(-x))

def tanh(x,diff=False):
    if diff:
        return (1 - x**2)/2
    return (1 - np.exp(-x))/(1 + np.exp(-x))

def relu(x,diff=False):
    tmp = np.zeros(x.shape)
    if diff:
        tmp[np.where(x>0)] = 1
        return tmp
    tmp[np.where(x>0)] = x[np.where(x>0)]
    return tmp

def leaky_relu(x,diff=False):
    if diff:
        tmp = np.ones(x.shape)
        tmp[np.where(x<=0)] = 0.01
        return tmp
    tmp = 0.01*x
    tmp[np.where(x>0)] = x[np.where(x>0)]
    return tmp

def linear(x,diff=False):
    if diff:
        return np.ones(x.shape)
    return x

# This dictionary will allows us later to access the activation functions
dict_act_fun = {"sigmoid":sigmoid,"tanh":tanh,"relu":relu,"leaky_relu":leaky_relu,"linear":linear} 

"""
Block Class

The way I see neural networks is as a series of blocks. Each Block is
representing the connection between 2 layers. The block will computer the input
and give an output.

"""
class BlockFCC:
    def __init__(self,n_inputs,n_outputs,act_fun="sigmoid"):
        self.weights  = np.random.random((n_inputs,n_outputs))
        self.bias = np.random.random(n_outputs)
        self.act_fun = dict_act_fun[act_fun]
    def forward(self,inp_data):
        self.inp = inp_data
        self.out = self.act_fun(inp_data.dot(self.weights) + self.bias)
        return self.out
    def backward(self,err):
        self.delta = err*self.act_fun(self.out,diff=True)
        return self.delta.dot(self.weights.T)
    def update(self,alpha=1):
        self.weights = self.weights - alpha*self.inp.T.dot(self.delta)
        self.bias = self.bias - alpha*self.delta.sum(axis=0)


class BlockSoftmax:
    def forward(self,inp_data):
        self.out = np.exp(inp_data)
        n = len(self.out)
        self.out = self.out/(self.out.sum(axis=1).reshape(n,1))
        return self.out
    def backward(self,err):
        tmp = self.out * err
        n = len(tmp)
        tmp = tmp - self.out*(tmp.sum(axis=1).reshape(n,1))
        return tmp
    def update(self,alpha=1):
        pass
    
"""
Loss Functions

Here I am defining some Loss Functions. The 2 more famous one (or at least the
ones I've learned so far) are the following 2.

- Mean Square Error
- Cross Entropy 
"""
def MSE(out,target): # Mean Square Error
    err = out - target
    loss = 0.5*err*err
    return loss.sum(),err

def CE(out,target): # Creoss Entropy Error
    loss = -target*np.log(out)
    err = -target/out
    return loss.sum(),err

# This dictionary will allow us to select the loss function we want
dict_loss_function = {"MSE":MSE,"CE":CE}

"""Model class

This class contains a series of blocks that made the actual neural network. This
class also contains some methods to train the neural network.

"""
class Model:
    def __init__(self,loss_function="MSE"):
        self.blocks = []
        self.loss_function = dict_loss_function[loss_function]
    def add(self,block):
        self.blocks.append(block)
    def forward(self,inp_data):
        self.out = inp_data
        for block in self.blocks:
            self.out = block.forward(self.out)
        return self.out
    def calculate_loss(self,target_data):
        return self.loss_function(self.out,target_data)
    def backward(self,err):
        err_tmp = err
        for block in reversed(self.blocks):
            err_tmp = block.backward(err_tmp)
    def update(self,alpha=1):
        for block in self.blocks:
            block.update(alpha=alpha)
    def set_loss_function(self,loss_function):
        self.loss_function = dict_loss_function[loss_function]
    def train(self,inp_data,target_data,maxIt = 1000,alpha=1):
        self.loss = []
        for i in range(maxIt):
            self.forward(inp_data)
            loss,err = self.calculate_loss(target_data)
            self.loss.append(loss)
            self.backward(err)
            self.update(alpha=alpha)
    def predict(self,inp_data):
        return self.forward(inp_data)

import sys 
    
if __name__ == "__main__":
    # Test: Classifying points
    if sys.argv[1] == "1":
        # Creating the neural network (model)
        model = Model()
        model.add(BlockFCC(2,5,act_fun="tanh"))
        model.add(BlockFCC(5,2,act_fun="relu"))
        model.add(BlockSoftmax())
        model.set_loss_function("CE")
        # Creating the training data
        N = 10 # Sample size
        x1 = 5 * np.random.random(N) - 2
        y1 = 5 * np.random.random(N) - 2
        C1 = [1,0] # Class 1
        C2 = [0,1] # Class 2
        inp_data = np.array([[i,j] for i,j in zip(x1,y1)])
        out_data = np.array([C1 if i<j else C2 for i,j in inp_data])
        # Training
        model.train(inp_data,out_data,maxIt=1000,alpha=0.1)
        # Predecting
        pred = model.predict(inp_data)
        for i,j in zip(pred,out_data):
            print(i,j)
        # Plotting the loos evolution while training
        plt.plot(model.loss)
        plt.show()
    # Test of FCC: XOR gate
    elif sys.argv[1] == "2":
        # Creating the neural network (model)
        model = Model()
        model.add(BlockFCC(2,3,act_fun="sigmoid"))
        model.add(BlockFCC(3,5,act_fun="tanh"))
        model.add(BlockFCC(5,2,act_fun="relu"))
        model.add(BlockSoftmax())
        model.set_loss_function("CE")
        # Creating the training data
        inp_data = np.array([[0,0],[1,0],[0,1],[1,1]])
        out_data = np.array([[1,0],[0,1],[0,1],[1,0]])
        # Training
        model.train(inp_data,out_data,alpha=0.1,maxIt=5000)
        # Predicting
        pred = model.predict(inp_data)
        print(pred)
        # Plotting the loss evolution while training 
        plt.plot(np.array(model.loss))
        plt.show()
