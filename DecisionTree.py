import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" This is my first attempt to create a Decision Tree Classifier, for discrete
data, usinf ID3 algorithm
"""

# We need to create a function that calculate the entropy of data
def calculate_entropy(y_data):
    unique_labels = np.unique(y_data)
    N = len(y_data)
    S = 0
    for l in unique_labels:
        tmp = np.where(y_data==l)
        m = len(tmp[0])
        p = m/N
        S -= p*np.log2(p)
    return S

# We also need a function to measure the Information Gain of an Attribute
def information_gain(x_data,y_data):
    unique_data = np.unique(x_data)
    N = len(x_data)
    IG = calculate_entropy(y_data)
    for d in unique_data:
        tmp = np.where(x_data == d)
        m   = len(tmp[0])
        p = m/N
        IG -= p*calculate_entropy(y_data[tmp])
    return IG

# We also need a class for the nodes of our tree
class TreeNode:
    def __init__(self):
        self.decision_idx = None
        self.data = None
        self.child = []
    def get_child_idx(self,x):
        return x[self.decision_idx]
    def predict_single(self,x):
        if len(self.child) == 0:
            return self.data
        print(x)
        child_idx = self.get_child_idx(x)
        return self.child[child_idx].predict_single(x)
    def predict(self,x):
        return np.array([self.predict_single(x0) for x0 in x])
    def print(self,depth = 0):
        print(" "*depth,self.data,self.decision_idx)
        for child in self.child:
            child.print(depth + 5)

# Finally we need a function that recursively creates the decision tree
def get_decision_tree(x_data,y_data,attr_idxs=[],start=True):
    root = TreeNode()
    S = calculate_entropy(y_data)
    if S == 0:
        root.data = y_data[0]
        return root
    if start:
        _,tmp = x_data.shape
        attr_idxs = np.array([i for i in range(tmp)])
        start = False
    if len(attr_idxs) == 0:
        root.data = y_data[0]
        return root
    info_gains = [information_gain(x_data[:,f],y_data) for f in attr_idxs]
    max_idx = info_gains.index(max(info_gains))
    new_atrr_idxs = np.delete(attr_idxs,max_idx)
    unique_data = np.unique(x_data[:,max_idx])
    root.decision_idx = max_idx
    # print("attr_idx: ",attr_idxs)
    # print("new_attr_idxs: ",new_atrr_idxs)
    for d in unique_data:
        tmp = np.where(x_data[:,max_idx] == d)
        root.child.append(get_decision_tree(x_data[tmp],y_data[tmp],new_atrr_idxs,start))
    return root

if __name__ == "__main__":
    # Reading sample data to classify
    data = pd.read_csv("data/data-test1.csv")
    y_train = data["Infected"].values
    x_train = data.drop("Infected",axis=1).values
    root= get_decision_tree(x_train,y_train)
    print(root)
    root.print()
    print(root.predict(x_train[3:6]))
    print(y_train[3:6])
