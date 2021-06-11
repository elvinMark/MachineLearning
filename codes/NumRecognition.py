import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
import PatternRecognition as pr
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import SimpleFCC as fcc
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    tmp = ds.load_digits()
    images, labels = tmp.images, tmp.target
    
    images = images.reshape(1797,64) 

    X_train, X_test, y_train, y_test = train_test_split(images,labels)

    if sys.argv[1] == "1":
        # Using Scikit learn Gauss Bayersian classifier
        gnb = GaussianNB()
        gnb.fit(X_train,y_train)
    
        pred = gnb.predict(X_test)
        print(confusion_matrix(pred,y_test))
        print(accuracy_score(pred,y_test))
    elif sys.argv[1] == '2':
        # Using my own gauss model classifier
        gauss = pr.get_model(X_train,y_train,10)
        pred =gauss.predict(X_test)
        print(confusion_matrix(pred,y_test))
        print(accuracy_score(pred,y_test))
    elif sys.argv[1] == "3":
        # Using Scikit Learn Multilayer Perceptron classifier
        clf = MLPClassifier(hidden_layer_sizes=[100,50],max_iter=300)
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        print(confusion_matrix(pred,y_test))
        print(accuracy_score(pred,y_test))
    elif sys.argv[1] == "4":
        # Using my own Fully Connected Neural Network
        n_samples,_ = X_train.shape 
        y_train1 = np.zeros([n_samples,10])
        for i,j in enumerate(y_train):
            y_train1[i][j] = 1
        x_train1 = X_train / np.max(X_train)
        model = fcc.Model()
        model.add(fcc.BlockFCC(64,32,act_fun="relu"))
        model.add(fcc.BlockFCC(32,10,act_fun="relu"))
        model.add(fcc.BlockSoftmax())
        model.set_loss_function("CE")
        model.train(x_train1,y_train1,maxIt=1000,alpha=0.0001)
        n_samples,_ = X_test.shape
        pred = model.predict(X_test)
        pred= np.round(pred)
        pred = pred.dot(np.array(range(10))).reshape(n_samples)
        print(confusion_matrix(pred,y_test))
        print(accuracy_score(pred,y_test))
    elif sys.argv[1] == "5":
        # classify using Decision Trees
        clf = DecisionTreeClassifier()
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        print(confusion_matrix(pred,y_test))
        print(accuracy_score(pred,y_test))
    elif sys.argv[1] == "6":
        # Classify using KNN algorithm
        clf = KNeighborsClassifier(n_neighbors=10,weights="distance")
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        print(confusion_matrix(pred,y_test))
        print(accuracy_score(pred,y_test))
