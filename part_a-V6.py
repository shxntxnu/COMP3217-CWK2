from sklearn import linear_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load training data
train_df = pd.read_csv("TrainingDataBinary.csv")

#load testing data
test_df = pd.read_csv("TestingDataBinary.csv")

X_train = train_df.iloc[:, :128]  # Extracting PMU measurement columns (1-116)
y_train = train_df[:, 128]  # Extracting the label column

# Clean the data by dividing by the maximum value
X_train = X_train / X_train.max()

# clean the data to replace infinite and large values with NaN
X_train[~np.isfinite(X_train)] = np.nan
X_train = np.nan_to_num(X_train)

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the logistic regression model
logistic = linear_model.LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)

# Get the number of samples
n_samples = len(X_train)

# Compute the training-test split
ratio = 0.3
X_train_split = X_train[:int(ratio * n_samples)]
y_train_split = y_train[:int(ratio * n_samples)]

# Load and preprocess the testing data
test_df = pd.read_csv('TestingDataBinary.csv')
X_test_final = test_df.iloc[:, 1:128]  # Extracting PMU measurement columns (1-116)

# Clean the testing data by dividing by the maximum value
X_test_final = X_test_final / X_train.max()

# clean the data to replace infinite and large values with NaN
X_test_final[~np.isfinite(X_test_final)] = np.nan
X_test_final = np.nan_to_num(X_test_final)

# Scale the features
X_test_final = scaler.transform(X_test_final)

# Predict labels for the testing data
predictions_final = logistic.predict(X_test_final)

# Compute the training-test split
X_test = X_train[int(ratio * n_samples):]
y_test = y_train[int(ratio * n_samples):]

# Compute the logistic regression accuracy
logistic_accuracy = logistic.score(X_test, y_test)
print("Logistic Regression accuracy: %f" % logistic_accuracy)

# Get predictions on the test data
predictions = logistic.predict(X_test)

# Compute the f1 score
f1 = f1_score(y_test, predictions, average='macro')
print("F1 score: %f" % f1)

# Compute the confusion matrix
cm = confusion_matrix(y_test, predictions, labels=logistic.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic.classes_)
disp.plot()
plt.show()

# Add predicted labels to testing data
test_df['marker'] = predictions_final

# Save the results to "TestingResultsBinary.csv"
test_df.to_csv('TestingResultsBinary.csv', index=False, header=train_df.columns)