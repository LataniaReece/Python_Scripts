#https://github.com/justmarkham/scikit-learn-videos/blob/master/07_cross_validation.ipynb

#Importing Datasets: 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import load_digits

#Loading the data:
digits = load_digits()

#Frequency counts in the target data 
unique, counts = np.unique(digits.target, return_counts = True)
np.asarray((unique, counts)).T

#Data Partition 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.3, random_state = 3)

#Logistic Regression
lr = LogisticRegression(solver='lbfgs', multi_class = 'multinomial', max_iter = 3000)
lr.fit(X_train, y_train)
lr.score(X_test, y_test) 

# You can get the same results above if you did the following:
# y_pred = lr.predict(X_train)
# from sklearn import metrics
# same as metrics.accuracy_score(y_test, y_pred)

#SVM
svm = SVC(gamma = 'auto')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

#Random Forest
rf = RandomForestClassifier(n_estimators = 40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

#KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

#Cross Validation 
from sklearn.model_selection import cross_val_score
#10-Fold cross_validation with K=5 for KNN ( the n_neighbors paramater)
knn = KNeighborsClassifier(n_neighbors = 5 )
scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = 'accuracy')
print(scores)

#use average accuracy as an estimate of out-of-sample accuracy 
print(scores.mean())

#Search for an optimal value of K for KNN
k_range = list(range(1,31))
k_scores = []
for k in k_range:
   knn = KNeighborsClassifier(n_neighbors = k)
   scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = 'accuracy')
   k_scores.append(scores.mean())
print(k_scores)

import matplotlib.pyplot as plt
%matplotlib qt5

#plot the value of K for KNN(x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

#Find max 
max_y = max(k_scores)  # Find the maximum y value
max_x = k_range[k_scores.index(max_y)]  # Find the x value corresponding to the maximum y value
print (max_x, max_y) # 1 is best, which means that this is probably overfitting 



#10 fold cross validation with the best KNN model 
knn = KNeighborsClassifier(n_neighbors = 20)
