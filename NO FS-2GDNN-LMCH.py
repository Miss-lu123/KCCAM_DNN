import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from keras.layers import Dense
from sklearn import metrics
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
from tensorflow.python.keras import layers
from keras.layers import preprocessing

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(tf.__version__)

# Read in data as pandas dataframe and display first 5 rows
data = pd.read_csv('selected_dia-lmch.csv')
print(data.head(5))

#One hot encoding
labels = data['Outcome']
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

# Remove the labels from the features
# axis 1 refers to the columns
data= data.drop('Outcome', axis = 1)

# Saving feature names for later use
data_list = list(data.columns)

# Convert to numpy array
data = np.array(data)

# Split the data into training and testing sets
train_features, val_test_features, train_labels, val_test_labels = train_test_split(data, dummy_y, test_size = 0.3,
                                                                           shuffle=True, random_state = 42)
X_val, X_test, Y_val, Y_test = train_test_split(val_test_features, val_test_labels, test_size=0.5, shuffle=True, random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)
print('Validation Features Shape:', X_val.shape)
print('Validation Label Shape:', Y_val.shape)

# normalizer = preprocessing.Normalization(axis=-1)
normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

#create model_2GDNN
def create_model(data):
    model = keras.Sequential([
        data,
        layers.Dense(12, activation='relu'),
        layers.Dense(12, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(3, activation="softmax")
    ])
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model

model1 = create_model(normalizer)
print(model1.summary())

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history=model1.fit(
    train_features, train_labels,
     validation_data=(X_val, Y_val),
     verbose=1, batch_size =30,epochs=200,callbacks=[tensorboard_callback])


#plot loss
def plot_loss(history):
  plt.plot(history.history['loss'], label='Train_loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Diabetes]')
  plt.legend()
  plt.grid(True)

plt.show()
plot_loss(history)



#plot accuracy
def plot_loss(history):
  plt.plot(history.history['accuracy'], label='Train_accuracy')
  plt.plot(history.history['val_accuracy'], label='val_accuracy')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy [Diabetes]')
  plt.legend()
  plt.grid(True)

plt.show()
plot_loss(history)

#Save model
#model1.save('DLmodelm.tf')
model_loaded = tf.keras.models.load_model('output/bestmodel.tf')

# Evaluation
# Prediction
prediction = np.around(model1.predict(X_test))
print(metrics.classification_report(Y_test,prediction, digits=8))

# Use the forest's predict method on the test data
predictions = model1.predict(X_test)

# Calculate mean absolute errors
score = model1.evaluate(X_test, Y_test,verbose=1)


# Compare Prediction
print("Train acc: " , model1.evaluate(train_features, train_labels))
print("Test acc: ", model1.evaluate(X_test, Y_test))


y = np.reshape(prediction,(150,3))
p  = np.argmax(y,axis=1)
Y=np.argmax(Y_test, axis=1)


# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
confusion = metrics.confusion_matrix(Y, p)
print(confusion)


# new_df = pd.DataFrame([[80, 75]])
# # We predict insulin
# prediction = model1.predict(new_df)
# print(prediction)


#Calculate metrics
y_true = Y
y_prediction = p
cnf_matrix = confusion_matrix(Y, p) #confusion_matrix(y_true, y_prediction)

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

import itertools

#Performance evaluation metrics
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


