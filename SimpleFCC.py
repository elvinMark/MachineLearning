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

dict_act_fun = {"sigmoid":sigmoid,"tanh":tanh,"relu":relu,"leaky_relu":leaky_relu,"linear":linear}

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
"""
def MSE(out,target): # Mean Square Error
    err = out - target
    loss = 0.5*err*err
    return loss.sum(),err

def CE(out,target): # Creoss Entropy Error
    loss = -target*np.log(out)
    err = -target/out
    return loss.sum(),err

dict_loss_function = {"MSE":MSE,"CE":CE}

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
if __name__ == "__main__":
    # Test of FCC: XOR gate
    model = Model()
    model.add(BlockFCC(2,3,act_fun="sigmoid"))
    model.add(BlockFCC(3,5,act_fun="tanh"))
    model.add(BlockFCC(5,2,act_fun="relu"))
    model.add(BlockSoftmax())
    model.set_loss_function("CE")
    inp_data = np.array([[0,0],[1,0],[0,1],[1,1]])
    out_data = np.array([[1,0],[0,1],[0,1],[1,0]])
    model.train(inp_data,out_data,alpha=0.1,maxIt=5000)
    pred = model.predict(inp_data)
    print(pred)
    plt.plot(np.array(model.loss))
    plt.show()
