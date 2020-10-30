import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DecisionTree:
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None
        self.class_type = None
    def predict_single(test):
        if self.left == None and self.right == None:
            return self.class_type
        idx = self.data["attribute_index"]
        threshold = self.data["threshold"]
        if test[idx] < threshold:
            return self.left.predict_single(test)
        else:
            return self.right.predict_single(test)
    def predict(self,test):
        return [self.predict_single(t) for t in test]

def calculate_entropy(y_train,class_type = 0):
    N = len(y_train)
    p = len(y_train[np.where(y_train == class_type)])/N
    n = 1 - p
    return -(p*np.log(p) + n*log(n))

def gain_entropy(x_train,y_train,attribute_index=0,class_type=0):
    return None

def get_decision_tree(x_train,y_train,attribute_indexes=None):
    root = DecisionTree()
    min_class = np.min(y_train)
    entropy = calculate_entropy(y_train,class_type=min_class)
    if entropy == 0:
        root.class_type = min_class
        return root
    if attribute_indexes == None:
        _,num_attributes = x_train.shape
        attribute_index = [i for i in range(num_attributes)]
    gain_entropies = [gain_entropy(x_train,y_train,attribute_index=i,class_type=min_class) for i in attribute_indexes]
    attribute_index = gain_entropies.index(max(gain_entropies))
    tmp = x_train[:,attribute_index]
    p = tmp[np.where(y_train == min_class)]
    n = tmp[np.where(y_train != min_class)]
    
    threshold = (np.max(p) + np.min(n))/2
    root.data = {"attribute_index":attribute_index,"threshold":threshold}
    left_x_train = x_train[np.where(y_train)]
    root.left = get_decision_tree(x_train)
    return None
