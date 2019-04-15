import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
type(iris)

#Prints data, feature names, target, target names 
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

#check shape of features and response 
print(iris.data.shape)
print(iris.target.shape)

#Store features and target into X and y respectively 
X = iris.data 
y = iris.target

#1 Import the class you plan to use 
#2 Instantiate the estimator 
#3 check the current paramters for the model (don't have to do)
#4 Fit the model
#5 predict 

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 1)
print(knn) 
knn.fit(X, y)

#prediction w/ 1 sample
sample = np.array([3,5,4,2]).reshape(1, -1)
knn.predict(sample)

#prediction with 2 samples 
X_new = [[3,5,4,3], [5,4,3,2]]
knn.predict(X_new)

#using a different value for K
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, y)
knn.predict(X_new)
