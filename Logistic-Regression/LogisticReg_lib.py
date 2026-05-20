import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dummy Data
# Features: Hours Studied (0 to 10)
X = np.array([[1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
# Target: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

# 1) Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2) Instantiate and train the model 
model = LogisticRegression()
model.fit(X_train, y_train)

# 3) Make predictions on test set (25% = 3 samples)
y_predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 4) Evaluate the performance 
model_accuracy = accuracy_score(y_test, y_predictions)

print("Predictions: \n", y_predictions)
print("Probabilities: \n", probabilities)
print("Accuracy: \n", model_accuracy)
print("Classification Report: \n", classification_report(y_test, y_predictions))
