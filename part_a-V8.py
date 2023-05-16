import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Data Preparation
training_data = pd.read_csv("TrainingDataBinary.csv")
testing_data = pd.read_csv("TestingDataBinary.csv")

X_train = training_data.iloc[:, :-1]  # Features (input variables) for training data
y_train = training_data.iloc[:, -1]   # Labels for training data

X_test = testing_data  # Features (input variables) for testing data

# Step 3: Model Selection and Training
model = LogisticRegression(random_state=0, max_iter=10000)
model.fit(X_train, y_train)

# Step 4: Compute the Test-training split
n_samples = len(X_train)
ratio = 0.7
X_train_split = X_train[:int(ratio * n_samples)]
y_train_split = y_train[:int(ratio * n_samples)]
X_test_split = X_train[int(ratio * n_samples):]
y_test_split = y_train[int(ratio * n_samples):]

# Step 4: Testing Data Prediction and Output
# predicted_labels = model.predict(X_test)
predicted_split_labels = model.predict(X_test_split)

# Create a DataFrame for the predicted labels
# testing_results = pd.DataFrame(X_test)
# testing_results['marker'] = predicted_labels
testing_split_results = pd.DataFrame(X_test_split)
testing_split_results['marker'] = predicted_split_labels

# Save the results to a CSV file
# testing_results.to_csv("TestingResultsBinary.csv", index=False)
testing_split_results.to_csv("TestingResultsBinary.csv", index=False)

# Calculate F1 score
# f1 = f1_score(y_train, model.predict(X_train))
f1_split = f1_score(y_train_split, model.predict(X_train_split))

# Calculate confusion matrix
# cm = confusion_matrix(y_train, model.predict(X_train))
cm_split = confusion_matrix(y_train_split, model.predict(X_train_split))

# Plot the confusion matrix
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.imshow(cm_split, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Abnormal'])
plt.yticks(tick_marks, ['Normal', 'Abnormal'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add labels to each cell
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(j, i, format(cm[i, j]), ha='center', va='center')
for i in range(cm_split.shape[0]):
    for j in range(cm_split.shape[1]):
        plt.text(j, i, format(cm_split[i, j]), ha='center', va='center')

plt.show()

print("F1 Score:", f1_split)