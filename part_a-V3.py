import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

#load training data
train_df = pd.read_csv("TrainingDataBinary.csv")

#load testing data
test_df = pd.read_csv("TestingDataBinary.csv")

# extract features and labels
X_train = train_df.iloc[:, :128].values
y_train = train_df.iloc[:, 128].values

X_test = test_df.iloc[:, :128].values
y_test = test_df.values

# clean the data to replace infinite and large values with NaN
X_train[~np.isfinite(X_train)] = np.nan
X_train = np.nan_to_num(X_train)

X_test[~np.isfinite(X_test)] = np.nan
X_test = np.nan_to_num(X_test)

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#print the training and test values
# print(X_train)
# print(y_train)

# print(X_test)

clf = linear_model.LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X_train, y_train)

#make predictions on the testing data
y_pred = clf.predict(X_test)

#get f1 score for predictions on the testing data
f1_score_test = f1_score(y_test, y_pred, average='weighted')
print(f1_score_test)