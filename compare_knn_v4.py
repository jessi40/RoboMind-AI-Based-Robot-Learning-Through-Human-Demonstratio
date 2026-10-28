
import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load existing model configuration
saved = joblib.load("models/action_model_v4.pkl")
features = saved["features"]
mins = saved["mins"]
maxs = saved["maxs"]
old_model = saved["model"]

# Load the original training and test datasets
train = pd.read_csv("data/processed/labeled_data_v4.csv")
test = pd.read_csv("data/processed/test_data_v4.csv")

X_train = train[features].copy()
y_train = train["action"]

X_test = test[features].copy()
y_test = test["action"]

# Use training limits for normalization
X_train = ((X_train - mins) / (maxs - mins)).fillna(0)
X_test = ((X_test - mins) / (maxs - mins)).fillna(0)

# Match the original training normalization
X_train = X_train.clip(0, 1)
X_test = X_test.clip(0, 1)

print("Original V4 accuracy: 71.76%")
print("\n===== KNN COMPARISON =====")

for k in [3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nk = {k}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    if k == 5:
        print("Classification report for k = 5:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))
