
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load V4 training data
df = pd.read_csv("data/processed/labeled_data_v4.csv")

# Select features
features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

X = df[features].values
y = df["action"].values

# Normalize features using training data only
mins = X.min(axis=0)
maxs = X.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1

X_scaled = (X - mins) / ranges

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

# Save model and normalization values
data = {
    "model": model,
    "features": features,
    "mins": mins,
    "maxs": maxs
}

with open("models/action_model_v4.pkl", "wb") as f:
    pickle.dump(data, f)

# Check training accuracy (not final test accuracy)
predictions = model.predict(X_scaled)
accuracy = accuracy_score(y, predictions)

print("V4 model trained successfully!")
print(f"Training accuracy: {accuracy * 100:.2f}%")
print("Saved: models/action_model_v4.pkl")

