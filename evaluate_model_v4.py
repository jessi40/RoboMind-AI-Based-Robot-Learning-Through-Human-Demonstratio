
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the existing V4 model
saved = joblib.load("models/action_model_v4.pkl")

model = saved["model"]
features = saved["features"]
mins = saved["mins"]
maxs = saved["maxs"]

# 2. Load the test dataset
df = pd.read_csv("data/processed/test_data_v4.csv")

# 3. Select input features and actual actions
X = df[features].copy()
y_true = df["action"]

# 4. Normalize using saved training limits
X = (X - mins) / (maxs - mins)

# Handle division by zero and missing values
X = X.replace([float("inf"), float("-inf")], 0)
X = X.fillna(0)

# 5. Predict actions using the existing model
y_pred = model.predict(X)

# 6. Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\n===== V4 MODEL EVALUATION =====")
print(f"Test samples: {len(y_true)}")
print(f"Correct predictions: {(y_true == y_pred).sum()}")
print(f"Incorrect predictions: {(y_true != y_pred).sum()}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 7. Precision, recall and F1-score
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_true,
    y_pred,
    labels=model.classes_,
    zero_division=0
))

# 8. Confusion matrix
print("===== CONFUSION MATRIX =====")
print("Class order:", list(model.classes_))
print(confusion_matrix(
    y_true,
    y_pred,
    labels=model.classes_
))
