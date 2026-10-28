
import csv
import random
import math
import pickle
from collections import Counter

random.seed(42)

# Load balanced dataset
with open("data/processed/balanced_data.csv", "r", newline="") as file:
    rows = list(csv.DictReader(file))

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

X = [[float(row[f]) for f in features] for row in rows]
y = [row["action"] for row in rows]

# Shuffle and split the data
indices = list(range(len(X)))
random.shuffle(indices)

split = int(len(indices) * 0.8)
train_idx = indices[:split]
test_idx = indices[split:]

X_train = [X[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
X_test = [X[i] for i in test_idx]
y_test = [y[i] for i in test_idx]

# Normalize using training data only
mins = [min(row[j] for row in X_train) for j in range(len(features))]
maxs = [max(row[j] for row in X_train) for j in range(len(features))]

def normalize(row):
    result = []
    for j, value in enumerate(row):
        span = maxs[j] - mins[j]
        result.append((value - mins[j]) / span if span else 0.0)
    return result

X_train = [normalize(row) for row in X_train]
X_test = [normalize(row) for row in X_test]

# Calculate KNN predictions
def predict(sample, k):
    distances = []

    for train_sample, label in zip(X_train, y_train):
        distance = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(sample, train_sample))
        )
        distances.append((distance, label))

    nearest = sorted(distances, key=lambda item: item[0])[:k]
    votes = Counter(label for _, label in nearest)

    # Break ties by nearest neighbor's label
    return max(
        votes,
        key=lambda label: (
            votes[label],
            -min(d for d, name in nearest if name == label)
        )
    )

# Test different K values
best_k = None
best_accuracy = -1

print("KNN accuracy comparison:")
for k in [1, 3, 5, 7, 9, 11]:
    predictions = [predict(row, k) for row in X_test]
    correct = sum(a == b for a, b in zip(y_test, predictions))
    accuracy = correct / len(y_test) * 100

    print(f"K={k}: {accuracy:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Train saved model on all balanced samples
all_mins = [min(row[j] for row in X) for j in range(len(features))]
all_maxs = [max(row[j] for row in X) for j in range(len(features))]

def normalize_all(row):
    return [
        (value - all_mins[j]) / (all_maxs[j] - all_mins[j])
        if all_maxs[j] != all_mins[j] else 0.0
        for j, value in enumerate(row)
    ]

X_all = [normalize_all(row) for row in X]

model = {
    "train_X": X_all,
    "train_y": y,
    "features": features,
    "mins": all_mins,
    "maxs": all_maxs,
    "k": best_k
}

with open("models/action_model_v2.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nBest K:", best_k)
print(f"Best test accuracy: {best_accuracy:.2f}%")
print("Saved model: models/action_model_v2.pkl")
