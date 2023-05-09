# These were the hyperparameters used and 
# Normalise data instead of replacing the values
# Use cross validation techniques (dimension reduction, grid-search)
# Accuracy: F1 score, confusion matrix, etc. "This was my training error, this was my cross-validation error."


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# load training data
train_df = pd.read_csv("TrainingDataBinary.csv")

# extract features and labels
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# clean the data to replace infinite and large values with NaN
X_train[~np.isfinite(X_train)] = np.nan
X_train = np.nan_to_num(X_train)

# get the number of features in the training data
num_features = X_train.shape[1]

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# train logistic regression model
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# load testing data
test_df = pd.read_csv("TestingDataBinary.csv")

# extract features
X_test = test_df.iloc[:, :num_features].values

# clean the data to replace infinite and large values with NaN
X_test[~np.isfinite(X_test)] = np.nan
X_test = np.nan_to_num(X_test)

# scale the features
X_test = scaler.transform(X_test)

# predict labels for testing data
y_pred = clf.predict(X_test)

# add predicted labels to testing data
test_df["marker"] = y_pred

# output results to file
test_df.to_csv("TestingResultsBinary.csv", index=False, header=train_df.columns)