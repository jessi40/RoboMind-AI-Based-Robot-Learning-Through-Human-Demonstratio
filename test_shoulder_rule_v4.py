
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report

# Load the existing model and test data
saved = joblib.load("models/action_model_v4.pkl")
model = saved["model"]
features = saved["features"]
mins = saved["mins"]
maxs = saved["maxs"]

df = pd.read_csv("data/processed/test_data_v4.csv")

X = df[features].copy()
y_true = df["action"]

# Normalize using the existing V4 training limits
X_scaled = ((X - mins) / (maxs - mins)).fillna(0).clip(0, 1)

# Original predictions
original_pred = model.predict(X_scaled)

print("Original accuracy:",
      round(accuracy_score(y_true, original_pred) * 100, 2), "%")

# Experimental post-processing rule
# Test several thresholds; do not save or replace the model.
for threshold in [30, 45, 60, 75]:
    adjusted_pred = original_pred.copy()

    condition = (
        (adjusted_pred == "raise_hand") &
        (X["right_shoulder_angle"] < threshold)
    )

    adjusted_pred[condition] = "lower_hand"

    print(f"\nThreshold: {threshold} degrees")
    print("Accuracy:",
          round(accuracy_score(y_true, adjusted_pred) * 100, 2), "%")

    print(classification_report(
        y_true,
        adjusted_pred,
        zero_division=0
    ))
