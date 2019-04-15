import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data 
y = iris.target

#logistic regression 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)
X_new = [[3,5,4,3], [5,4,3,2]]
logreg.predict(X_new)