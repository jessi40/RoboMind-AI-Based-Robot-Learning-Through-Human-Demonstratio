
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


# Load training data
df = pd.read_csv("data/processed/labeled_data_v4.csv")

saved = joblib.load("models/action_model_v4.pkl")
features = saved["features"]
mins = saved["mins"]
maxs = saved["maxs"]

X = df[features].copy()
y = df["action"]

# Use the same normalization limits as V4
X = ((X - mins) / (maxs - mins)).fillna(0).clip(0, 1)

# Keep separate training and validation portions
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train a temporary model only
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

original_pred = model.predict(X_val)

print("Original validation accuracy:",
      round(accuracy_score(y_val, original_pred) * 100, 2), "%")

# Test the selected rule
adjusted_pred = original_pred.copy()

condition = (
    (adjusted_pred == "raise_hand") &
    (df.loc[X_val.index, "right_shoulder_angle"].to_numpy() < 30)
)

adjusted_pred[condition] = "lower_hand"

print("\nRule-adjusted validation accuracy:",
      round(accuracy_score(y_val, adjusted_pred) * 100, 2), "%")

print("\nValidation classification report:")
print(classification_report(y_val, adjusted_pred, zero_division=0))

print("\nOriginal KNN classification report:")
print(classification_report(y_val, original_pred, zero_division=0))

print("\nOriginal KNN confusion matrix:")
print(confusion_matrix(
    y_val,
    original_pred,
    labels=["bend_arm", "lower_hand", "raise_hand"]
))

