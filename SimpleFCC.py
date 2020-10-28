"""
Simple Fully Connected Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Activation Functions
"""
def sigmoid(x,diff=False):
    if diff:
        x*(1-x)
    return 1/(1 + np.exp(-x))

def tanh(x,diff=False):
    if diff:
        (1 - x**2)/2
    return (1 - np.exp(-x))/(1 + np.exp(-x))

dict_act_fun = {"sigmoid":sigmoid,"tanh":tanh}

class Block:
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


"""
Loss Functions
"""
def MSE(out,target):
    err = out - target
    loss = 0.5*err*err
    return loss.sum(),err

class Model:
    def __init__(self):
        self.blocks = []
    def add(self,block):
        self.blocks.append(block)
    def forward(self,inp_data):
        self.out = inp_data
        for block in self.blocks:
            self.out = block.forward(self.out)
        return self.out
    def calculate_loss(self,target_data):
        return MSE(self.out,target_data)
    def backward(self,err):
        err_tmp = err
        for block in reversed(self.blocks):
            err_tmp = block.backward(err_tmp)
    def update(self,alpha=1):
        for block in self.blocks:
            block.update(alpha=alpha)
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
if __name__ == "__main__":
    # Test of FCC: XOR gate
    model = Model()
    model.add(Block(2,3))
    model.add(Block(3,2))
    inp_data = np.array([[0,0],[1,0],[0,1],[1,1]])
    out_data = np.array([[1,0],[0,1],[0,1],[1,0]])
    model.train(inp_data,out_data)
    pred = model.predict(inp_data)
    print(pred)
