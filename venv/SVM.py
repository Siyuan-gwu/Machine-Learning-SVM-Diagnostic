import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import time

data = pd.read_csv('wdbc.data')
# data.info()
# data.columns
# replace 'M' and 'B' with 1 and 0
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
print (data['diagnosis'])
# dataset[1] = dataset[1].map({'M' : 1, 'B' : 0})
# print (dataset[1])
# drop the column 0, which contains 'id' (useless)
data.drop('id', axis=1, inplace=True)
print (data.head(5))
# dataset.drop(columns=0, axis=1, inplace=True)
feature = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']
# visualization
# data[feature].hist(bins=50, figsize=(20, 15))
# plt.show()

from sklearn.model_selection import train_test_split
train, test = train_test_split(data,test_size=0.3,train_size=0.7)
feature = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']
# 2, 3, 6, 7, 10, 11
print (train.shape)
train_X = train[feature]
train_y = train['diagnosis']
test_X = test[feature]
test_y = test['diagnosis']
print (train_X.head(5))
# min-max normalization
def MaxMinNormalization(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
train_X = MaxMinNormalization(train_X)
test_X = MaxMinNormalization(test_X)

print (train_X)
print (train_y.shape)

def confusion_matrix(y_true, y_pred):
    matrix = np.zeros([2, 2])
    for y_true, y_pred in zip(y_true,y_pred):
        if y_true == 1 and y_pred == 1:
           matrix[0][0] += 1
        if y_true == 0 and y_pred == 1:
           matrix[0][1] += 1
        if y_true == 0 and y_pred == 0:
           matrix[1][1] += 1
        if y_true == 1 and y_pred == 0:
           matrix[1][0] += 1

    return matrix

# Training...
# ------------------------
print ("Training...")
model = svm.SVC()
start = time.thread_time()
model.fit(train_X, train_y)
## step 3: testing
print ("Testing...")
prediction = model.predict(test_X)
end = time.thread_time()
print ('Time used: ', (end - start))
## step 4: show the result
print ("show the result...")
errorCount = 0
for y_res, y_predict in zip(test_y, prediction):
    if y_res != y_predict:
        errorCount += 1
print ('The classify accuracy is: ', (len(test_y) - errorCount) / len(test_y))
c_matrix = confusion_matrix(test_y, prediction)
print (c_matrix)



