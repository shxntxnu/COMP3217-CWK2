import pandas as pd
import numpy as np
from sklearn import linear_model, svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
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

# perform 5-fold cross-validation on the training data and compute the mean F1 score
f1_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted')
print("Cross-validation F1 scores: ", f1_scores)
print("Mean F1 score: ", np.mean(f1_scores))

# train the model on the full training data
clf.fit(X_train, y_train)

#make predictions on the testing data
y_pred = clf.predict(X_test)

#get f1 score for predictions on the testing data
# The 'macro' average calculates the F1 score for each class separately, and then takes the unweighted average across all classes. In other words, it treats each class equally and gives the same weight to each class, regardless of its frequency in the data.
# The 'weighted' average calculates the F1 score for each class separately, and then takes the weighted average across all classes, where the weights are proportional to the number of samples in each class. In other words, it gives more weight to the classes with more samples and less weight to the classes with fewer samples.
f1_score_test = f1_score(test_df.iloc[:, -1].values, y_pred, average='weighted')
print("Weighted F1 score: ", f1_score_test)

f1_score_test = f1_score(test_df.iloc[:, -1].values, y_pred, average='macro')
print("Macro F1 score: ", f1_score_test)

#get accuracy for the implementation
accuracy_test = metrics.accuracy_score(test_df.iloc[:, -1], y_pred)
print("Accuracy test: ", accuracy_test)

#get confusion matrix
cm = confusion_matrix(test_df.iloc[:, -1].values, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# add predicted labels to testing data
test_df["marker"] = y_pred

# output results to file
test_df.to_csv("TestingResultsBinary.csv", index=False, header=train_df.columns)