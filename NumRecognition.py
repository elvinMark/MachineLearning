import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
import PatternRecognition as pr

tmp = ds.load_digits()
images, labels = tmp.images, tmp.target

images = images.reshape(1797,64) 

X_train, X_test, y_train, y_test = train_test_split(images,labels)


gnb = GaussianNB()
gnb.fit(X_train,y_train)

pred = gnb.predict(X_test)
print(confusion_matrix(pred,y_test))
print(accuracy_score(pred,y_test))

gauss = pr.get_model(X_train,y_train,10)
pred =gauss.predict(X_test)
print(confusion_matrix(pred,y_test))
print(accuracy_score(pred,y_test))
