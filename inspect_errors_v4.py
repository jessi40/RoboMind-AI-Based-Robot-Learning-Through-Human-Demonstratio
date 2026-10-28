
import pandas as pd
import joblib

from collections import Counter

# Load existing V4 model
saved = joblib.load("models/action_model_v4.pkl")
model = saved["model"]
features = saved["features"]
mins = saved["mins"]
maxs = saved["maxs"]

# Load held-out test data
df = pd.read_csv("data/processed/test_data_v4.csv")

X = df[features].copy()
y_true = df["action"]

# Apply the saved V4 normalization
X = ((X - mins) / (maxs - mins)).fillna(0).clip(0, 1)

# Get predictions
y_pred = model.predict(X)

results = df[["time", "action"]].copy()
results["predicted_action"] = y_pred
results["correct"] = results["action"] == results["predicted_action"]

print("===== OVERALL ERRORS =====")
print("Total test frames:", len(results))
print("Incorrect frames:", (~results["correct"]).sum())

print("\n===== ACTUAL VS PREDICTED =====")
print(pd.crosstab(
    results["action"],
    results["predicted_action"]
))

print("\n===== LOWER_HAND MISCLASSIFICATIONS =====")
errors = results[
    (results["action"] == "lower_hand") &
    (~results["correct"])
]

print("Misclassified lower_hand frames:", len(errors))
print("\nMost common incorrect predictions:")
print(errors["predicted_action"].value_counts())

print("\nFirst 15 misclassified examples:")
print(errors.head(15).to_string(index=False))
