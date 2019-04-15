#url: https://youtu.be/3ZWuPVWq7p4?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',
                   index_col=0)
data.head()

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales',
             kind = "reg")

#Create X and y 
#method1 - X
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
#method2 - X
X = data[['TV', 'radio', 'newspaper']]
#method1 - y
y = data['sales']
#method2 - y
y = data.sales
y.head()

#Data Partition - default split = 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

#Model 
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#print intercept and coefficients
linreg.intercept_
linreg.coef_

#pair the feature names with the coef 
list(zip(feature_cols, linreg.coef_))

#eval metrics 
y_pred = linreg.predict(X_test)
metrics.mean_absolute_error(y_test, y_pred) #MAE
metrics.mean_squared_error(y_test, y_pred) #MSE
np.sqrt(metrics.mean_squared_error(y_test, y_pred)) #RMSE



