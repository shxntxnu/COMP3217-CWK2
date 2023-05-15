from sklearn import linear_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the training data
training_data = pd.read_csv('TrainingDataBinary.csv')
X_train = training_data.iloc[:, 1:128]  # Extracting PMU measurement columns (1-116)
y_train = training_data['marker']  # Extracting the label column

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

# Load and preprocess the testing data
testing_data = pd.read_csv('TestingDataBinary.csv')
X_test = testing_data.iloc[:, 1:128]  # Extracting PMU measurement columns (1-116)

# Clean the testing data by dividing by the maximum value
X_test = X_test / X_train.max()

# clean the data to replace infinite and large values with NaN
X_test[~np.isfinite(X_test)] = np.nan
X_test = np.nan_to_num(X_test)

# scale the features
X_test = scaler.transform(X_test)

# Predict labels for the testing data
predictions = logistic.predict(X_test)

# Calculate f1 score
f1 = f1_score(testing_data.iloc[:, -1].values, predictions, average='macro')
print("F1 Score: %f" % f1)

# Calculate confusion matrix
cm = confusion_matrix(testing_data.iloc[:, -1].values, predictions, labels=logistic.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic.classes_)
disp.plot()
plt.show()

# Create a new DataFrame with the predicted labels
results = pd.DataFrame()
results['marker'] = predictions

# Save the results to "TestingResultsBinary.csv"
results.to_csv('TestingResultsBinary.csv', index=False)
