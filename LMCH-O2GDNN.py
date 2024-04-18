#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from keras.layers import Dense
from sklearn import metrics
from tensorflow import keras
from tensorflow.python.keras import layers
from keras.layers import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
# multi-class classification with Keras
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
print(tf.__version__)

# Read in data as pandas dataframe and display first 5 rows
data = pd.read_csv('selected_sklearn3.csv')
print(data.head(5))
# 启动计时器
start_time = time.time()

from keras.utils.np_utils import to_categorical
labels = data['Outcome']

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

# Remove the labels from the features
data= data.drop('Outcome', axis = 1)

# Saving feature names for later use
data_list = list(data.columns)

# Convert to numpy array
data = np.array(data)

# Split the data into training and testing sets
train_features, val_test_features, train_labels, val_test_labels = train_test_split(data, dummy_y, test_size=0.3,
                                                                           random_state =35)
X_val, X_test, Y_val, Y_test = train_test_split(val_test_features, val_test_labels, test_size=0.5)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)
print('Validation Features Shape:', X_val.shape)
print('Validation Label Shape:', Y_val.shape)

def create_model():
    # normalizer = layers.experimental.preprocessing.Normalization(axis=-1)
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    model = keras.Sequential([
      normalizer,# 对输入数据进行标准化或归一化，使每个特征的值域大致相等
      layers.Dense(12, activation='relu'),
      layers.Dense(12, activation='relu'),
      layers.Dense(24, activation='relu'),
      layers.Dense(24, activation='relu'),
      layers.Dense(3, activation="softmax")])
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
# create model
model = KerasClassifier(build_fn=create_model, verbose=2)
model.fit(train_features, train_labels)
# define the grid search parameters
# batch_size = [1, 5, 10, 20]
# epochs = [10, 50, 100, 200]
batch_size = [5]
epochs = [200]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#param_grid = dict(optimizer=optimizer)
param_grid = dict(batch_size=batch_size, epochs=epochs)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=12)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=10)
grid_result = grid.fit(train_features, train_labels, validation_data=(X_val, Y_val))

cv_scores = grid_result.cv_results_
# best_params_=grid.best_estimator_.get_params()
# print(best_params_)

print(grid_result.best_params_)

# 停止计时器
end_time = time.time()

# 计算并输出训练时间
elapsed_time = end_time - start_time
print('Training time:', elapsed_time, 'seconds')

##printing all the parameters along with their scores
for mean_score, params in zip(cv_scores['mean_test_score'], cv_scores["params"]):
    print(mean_score, params)


y_pred = grid_result.predict(X_test)
Y_test = np.argmax(Y_test, axis=1)
print(metrics.classification_report(Y_test,y_pred, digits=8))

# Compare Prediction 
print("Train acc: " , grid_result.score(train_features, train_labels))
print("Test acc: ", grid_result.score(X_test, Y_test))


