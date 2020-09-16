'''
Using the code from this site for the base since it works,
However there is an error with the 'fit()'.

https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
'''


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import pdb

import os
print(os.listdir(r"."))         #print out what's in the file/directory

train = pd.read_csv(r"Train_featMat.csv")
test = pd.read_csv(r"featMatVersion2_5.csv")

print("test")


#print
train.head()

#describe the column
train.describe()

#print out the lables/column and there datatype
train.dtypes


#prints the selected columns
#train_knn = train[['User_ID', 'Doc_ID', 'Phone_ID', 'Ratio dist and length of trajectory', 'Average velocity', 'Phone orientation']]
train_knn = train[['User_ID', 'Doc_ID']]
train_knn.head()

#display the dtype of the select column
train_knn.dtypes


'''
Choosing a specific columns to use for now
'''
#prints the selected columns
#train_knn = train[['User_ID', 'Doc_ID', 'Phone_ID', 'Ratio dist and length of trajectory', 'Average velocity', 'Phone orientation']]
train_knn = train[['User_ID', 'Doc_ID']]
train_knn.head()


#The X variable contains the first five columns of the dataset (i.e. attributes) while y contains the labels.
X = train_knn.iloc[:, :-1].values
y = train_knn.iloc[:, :].values


#Train and test split
#80% train, 20% test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#X_train=X_train.astype('int')
#X_test=X_test.astype('int')

#Error on "classifier.fit()
#Training and Prediction
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

print(classifier)
