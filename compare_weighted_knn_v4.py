
import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the existing V4 model configuration
saved = joblib.load("models/action_model_v4.pkl")
features = saved["features"]
mins = saved["mins"]
maxs = saved["maxs"]

# Load the same training and test datasets
train = pd.read_csv("data/processed/labeled_data_v4.csv")
test = pd.read_csv("data/processed/test_data_v4.csv")

X_train = train[features].copy()
y_train = train["action"]

X_test = test[features].copy()
y_test = test["action"]

# Apply the saved V4 normalization
X_train = ((X_train - mins) / (maxs - mins)).fillna(0).clip(0, 1)
X_test = ((X_test - mins) / (maxs - mins)).fillna(0).clip(0, 1)

# Compare ordinary KNN with distance-weighted KNN
for weights in ["uniform", "distance"]:
    model = KNeighborsClassifier(n_neighbors=5, weights=weights)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\n===== KNN weights={weights} =====")
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print(classification_report(y_test, predictions, zero_division=0))
