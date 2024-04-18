#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras import layers
from keras.layers import preprocessing
import matplotlib.pyplot as plt

print(tf.__version__)

from sklearn.preprocessing import LabelEncoder
# Read in data as pandas dataframe and display first 5 rows
data = pd.read_csv('selfeatures-nonzero.csv')
print(data.head(5))

# Use numpy to convert to arrays
import numpy as np
labels = data['Outcome']

# Remove the labels from the features
# axis 1 refers to the columns
data= data.drop('Outcome', axis = 1)

# Saving feature names for later use
data_list = list(data.columns)

# Convert to numpy array
data = np.array(data)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, val_test_features, train_labels, val_test_labels = train_test_split(data, labels, shuffle=True, test_size = 0.3,
                                                                           random_state =42)
X_val, X_test, Y_val, Y_test = train_test_split(val_test_features, val_test_labels, shuffle=True, test_size=0.5, random_state=0)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)
print('Validation Features Shape:', X_val.shape)
print('Validation Label Shape:', Y_val.shape)


# main model designed

from numpy import mean
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from numpy import std
from numpy import mean

# generate dataset
model = SVC(kernel='rbf', C=1000)
# evaluate the model
print(model.fit(train_features, train_labels))

# Compare Prediction 
print("Train acc: ", model.score(train_features, train_labels))
print("Test acc: ", model.score(X_test, Y_test))

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
# Prediction
y_pred = np.around(model.predict(X_test))
print(metrics.classification_report(Y_test,y_pred, digits=8))


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(Y_test, y_pred)
print(mat)


# Optimization

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import sklearn.svm as svm
# create model
model = SVC(kernel='rbf')
# define the grid search parameters
# Set the parameters by cross-validation
#param_grid = [{"C": [1, 10, 100, 1000], "gamma": [1e-3, 1e-4]}]
# param_grid = [{'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}]
param_grid = [{'C': [1,10,100,1000], 'gamma': [0.01,0.001],'kernel': ['rbf']}]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=cv)
print(grid)

grid_result = grid.fit(train_features, train_labels)
print(grid_result)
best_parameters=grid.best_estimator_.get_params()
print('最佳参数',best_parameters)

# grid_result.best_params_

cv_scores = grid_result.cv_results_
print(cv_scores)
##printing all the parameters along with their scores

for mean_score, params in zip(cv_scores['mean_test_score'], cv_scores["params"]):
    print(mean_score, params)
# mean_score=grid_result.cv_scores['mean_test_score']
# params=grid_result.cv_scores["params"]
#
# for mean_score, params in zip(mean_score, params):
#     print(mean_score, params)

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
# Prediction
y_pred = np.around(grid_result.predict(X_test))
print(metrics.classification_report(Y_test,y_pred, digits=8))

# Compare Prediction
print("Train acc: " , grid_result.score(train_features, train_labels))
print("Test acc: ", grid_result.score(X_test, Y_test))

