#https://github.com/justmarkham/scikit-learn-videos/blob/master/08_grid_search.ipynb

from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#read the iris data 
iris = load_iris()

#create X (features) and y (response)
X = iris.data
y = iris.target

##10 fold cross validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors = 5)
#scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
#print(scores)
#
#print(scores.mean())
#
##search for an optimal value of K for KNN
#k_range = list(range(1,31))
#k_scores = []
#for k in k_range:
#   knn = KNeighborsClassifier(n_neighbors = k)
#   scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
#   k_scores.append(scores.mean())
#print(k_scores)
#
##plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
#plt.plot(k_range, k_scores)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Cross-Validated Accuracy')

#More efficient parameter tuning using GridSearchCV--------------
from sklearn.model_selection import GridSearchCV

#define the parameter values that should be searched 
k_range = list(range(1,31))
print(k_range)

#create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors = k_range)
print(param_grid)

#instantiate the grid 
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy', return_train_score = False)

#You can set n_jobs = -1 to run computations in parallel (if supported by your computer and OS)

#Fit the grid with the data 
grid.fit(X, y)

#View the results as a pandas DataFrame
import pandas as pd
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# examine the first result
print(grid.cv_results_['params'][0])
print(grid.cv_results_['mean_test_score'][0])

# print the array of mean scores only
grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)

# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

#Searching multiple parameters simultaneously -------------------

# define the parameter values that should be searched
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)

# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)

pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

# examine the best model
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)

#Using the best parameters to make predictions 

# train your model using all data and the best known parameters
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)

# make a prediction on out-of-sample data
knn.predict([[3, 5, 4, 2]])

#ALTERNATE METHOD TO ABOVE, JUST USE GRIDSEARCH
# shortcut: GridSearchCV automatically refits the best model using all of the data
grid.predict([[3, 5, 4, 2]])

#Reducing computational budget using RandomizedSearchCV
# GridSearch searches many different parameters at once may be computationally infeasible
#RandomizedSearchCV searches a subset of the parameters, and you control the computational "budget"

from sklearn.model_selection import RandomizedSearchCV

# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)
