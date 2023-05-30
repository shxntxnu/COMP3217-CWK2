# LOGISTIC REGRESSION
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, f1_score
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt

# # Step 1: Data Preparation
# training_data = pd.read_csv("TrainingDataMulti.csv")
# testing_data = pd.read_csv("TestingDataMulti.csv")

# # Step 2: Extract features
# X_train = training_data.iloc[:, :-1] # Features (input variables) for training data
# y_train = training_data.iloc[:, -1] # Labels for training data

# X_test = testing_data # Features (input variables) for testing data

# # Step 3: Model Selection and Hyperparameter Tuning
# model = LogisticRegression(random_state=0, max_iter=10000)

# # Perform cross-validation for hyperparameter tuning
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')

# # Print the cross-validation scores
# print("Cross-Validation Scores:", scores)

# # Fit the model on the entire training data
# model.fit(X_train, y_train)

# # Step 4: Testing Data Prediction and Output
# predicted_labels = model.predict(X_test)

# # Create a DataFrame for the predicted labels
# testing_results = pd.DataFrame(testing_data)
# testing_results['label'] = predicted_labels

# # Save the results to a CSV file
# testing_results.to_csv("TestingResultsMulti.csv", index=False)

# # Calculate F1 score
# f1 = f1_score(y_train, model.predict(X_train), average='macro')

# # Calculate confusion matrix
# cm = confusion_matrix(y_train, model.predict(X_train))

# # Plot the confusion matrix
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# tick_marks = np.arange(3)
# plt.xticks(tick_marks, ['Normal (0)', 'Data Injection (1)', 'Command Injection (2)'])
# plt.yticks(tick_marks, ['Normal (0)', 'Data Injection (1)', 'Command Injection (2)'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')

# # Add labels to each cell
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(j, i, format(cm[i, j]), ha='center', va='center')

# plt.show()

# print("F1 Score:", f1)

# RANDOM FOREST
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Step 1: Data Preparation
training_data = pd.read_csv("TrainingDataMulti.csv")
testing_data = pd.read_csv("TestingDataMulti.csv")

# Step 2: Extract features
X_train = training_data.iloc[:, :-1] # Features (input variables) for training data
y_train = training_data.iloc[:, -1] # Labels for training data

X_test = testing_data # Features (input variables) for testing data

# Step 3: Model Selection and Hyperparameter Tuning
model = DecisionTreeClassifier(random_state=0)

# Perform cross-validation for hyperparameter tuning
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)

# Fit the model on the entire training data
model.fit(X_train, y_train)

# Step 4: Testing Data Prediction and Output
predicted_labels = model.predict(X_test)

# Create a DataFrame for the predicted labels
testing_results = pd.DataFrame(testing_data)
testing_results['label'] = predicted_labels

# Save the results to a CSV file
testing_results.to_csv("TestingResultsMulti.csv", index=False)

# Calculate F1 score
f1 = f1_score(y_train, model.predict(X_train), average='macro')

# Calculate confusion matrix
cm = confusion_matrix(y_train, model.predict(X_train))

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Normal (0)', 'Data Injection (1)', 'Command Injection (2)'])
plt.yticks(tick_marks, ['Normal (0)', 'Data Injection (1)', 'Command Injection (2)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add labels to each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j]), ha='center', va='center')

plt.show()

print("F1 Score:", f1)