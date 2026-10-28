
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Load V4 training dataset
df = pd.read_csv("data/processed/labeled_data_v4.csv")

# Use only left-arm features
features = [
    "left_elbow_angle",
    "left_shoulder_angle"
]

X = df[features].values
y = df["action"].values

# Normalize the features
mins = X.min(axis=0)
maxs = X.max(axis=0)

ranges = maxs - mins
ranges[ranges == 0] = 1

X_scaled = (X - mins) / ranges

# Train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

# Save as a new V5 model
saved_data = {
    "model": model,
    "features": features,
    "mins": mins,
    "maxs": maxs
}

with open("models/action_model_v5.pkl", "wb") as f:
    pickle.dump(saved_data, f)

print("V5 model trained successfully!")
print("Features used:", features)
print("Saved: models/action_model_v5.pkl")

