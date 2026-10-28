
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
train = pd.read_csv("data/processed/labeled_data_v4.csv")
test = pd.read_csv("data/processed/test_data_v4.csv")

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

X_train = train[features]
y_train = train["action"]

X_test = test[features]
y_test = test["action"]

# Test Decision Tree
dt = DecisionTreeClassifier(
    max_depth=8,
    random_state=42
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("===== DECISION TREE =====")
print("Accuracy:", round(accuracy_score(y_test, dt_pred) * 100, 2), "%")
print(classification_report(y_test, dt_pred, zero_division=0))

# Test Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", round(accuracy_score(y_test, rf_pred) * 100, 2), "%")
print(classification_report(y_test, rf_pred, zero_division=0))
