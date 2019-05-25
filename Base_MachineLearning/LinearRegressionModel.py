#Simple Linear Regression 

#Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#importing the dataset
#Chemical Manufactoring DataSet, predicting yield with other predictors
df = pd.read_csv(r'C:\Users\reece\Desktop\Python_Scripts\LearningPython\ChemicalManufacturingProcessData.csv')

#Viewing data 
df.dtypes
df.columns
df.columns.get_loc('Yield') #0
X = df.drop(['Yield'], axis = 1)
y=df.Yield

#Their way of creating X and y, I prefer the way above
#    X = df.iloc[:, 1:].values
#    y = df.iloc[:, 0].values


#Filling in missings with median values for columns
X.isna().sum()

features = X.dtypes.index
feature_type = X.dtypes

X.dtypes.value_counts()
dtypes_dictionary = dict(zip(features, feature_type))


replacements = X.apply(np.nanmedian, 'rows') #Rememeber going down the rows gets you the median for the columns 
replacement_dictionary = dict(zip(features, replacements))

for feature, feature_type in dtypes_dictionary.items():
    if feature_type == 'int64':
        X[feature] = X[feature].fillna(replacement_dictionary[feature])
    else:
        X[feature] = X[feature].fillna(math.floor(replacement_dictionary[feature]))
        
#Data Partition 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
                                                    random_state = 0)

#Feature Scaling 
""" from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Simple Linear Regressoin to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set Results 
y_pred = regressor.predict(X_test)

#Viewing coefficients and intercepts 
print('coefficients: ', regressor.coef_)
print('intercept: ', regressor.intercept_)


#Visualizing the Training set results - residuals

#predicted vs actual
plt.scatter(regressor.predict(X_train), y_train, color = 'red')
plt.title('Predicted Vs Actual')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Predicted vs Residuals
residuals = y_train - regressor.predict(X_train) 
plt.scatter(regressor.predict(X_train), residuals, color = 'red')
plt.axhline(0, 0, 1)
plt.title('Predicted Vs Residuals')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()














