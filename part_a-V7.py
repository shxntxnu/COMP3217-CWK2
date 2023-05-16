import pandas as pd
import numpy as np
from sklearn import linear_model, svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
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

# Clean the data by dividing by the maximum value
X_train = X_train / X_train.max()
X_test = X_test / X_train.max()

# clean the data to replace infinite and large values with NaN
X_train[~np.isfinite(X_train)] = np.nan
X_train = np.nan_to_num(X_train)

X_test[~np.isfinite(X_test)] = np.nan
X_test = np.nan_to_num(X_test)

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train the logistic regression model
clf = linear_model.LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X_train, y_train)

# Get the number of samples
n_samples = len(X_train)

# Compute the training-test split
ratio = 0.3
X_train_split = X_train[:int(ratio * n_samples)]
y_train_split = y_train[:int(ratio * n_samples)]
X_test = X_train[int(ratio * n_samples):]
y_test = y_train[int(ratio * n_samples):]

#print the training and test values
# print(X_train)
# print(y_train)

# print(X_test)

# Compute the logistic regression accuracy
logistic_accuracy = clf.score(X_test, y_test)
print("Logistic Regression accuracy: %f" % logistic_accuracy)

# Get predictions on the test data
predictions = clf.predict(X_test)

# Compute the f1 score
f1 = f1_score(y_test, predictions, average='macro')
print("F1 score: %f" % f1)

# train the model on the full training data
clf.fit(X_train, y_train)

#make predictions on the testing data
y_pred = clf.predict(X_test)

#get confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# add predicted labels to testing data
test_df['marker'] = y_pred

# output results to file
test_df.to_csv("TestingResultsBinary.csv", index=False, header=train_df.columns)