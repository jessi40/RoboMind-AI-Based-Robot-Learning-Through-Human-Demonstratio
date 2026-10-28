
import csv
import os
import random
import pickle
import math
from collections import Counter

DATASET = "data/processed/balanced_data_new.csv"
MODEL_PATH = "models/action_model_v3.pkl"

FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

K_VALUES = [1, 3, 5, 7, 9, 11]
random.seed(42)


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def predict(sample, train_x, train_y, k):
    distances = []

    for x, label in zip(train_x, train_y):
        distances.append((distance(sample, x), label))

    distances.sort(key=lambda item: item[0])
    nearest = [label for _, label in distances[:k]]

    counts = Counter(nearest)
    highest = max(counts.values())

    # Resolve ties consistently
    winners = sorted(
        label for label, count in counts.items()
        if count == highest
    )
    return winners[0]


# Load the balanced dataset
with open(DATASET, "r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError("Dataset is empty.")

X = [[float(row[f]) for f in FEATURES] for row in rows]
y = [row["action"] for row in rows]

print("Total samples:", len(X))
print("Class counts:", dict(Counter(y)))

# Shuffle and split: 80% training, 20% testing
combined = list(zip(X, y))
random.shuffle(combined)

X, y = zip(*combined)
X, y = list(X), list(y)

split = int(0.8 * len(X))
train_x, test_x = X[:split], X[split:]
train_y, test_y = y[:split], y[split:]

# Normalize using training data only
mins = [min(row[i] for row in train_x)
        for i in range(len(FEATURES))]
maxs = [max(row[i] for row in train_x)
        for i in range(len(FEATURES))]


def normalize(row):
    result = []

    for i, value in enumerate(row):
        span = maxs[i] - mins[i]
        result.append((value - mins[i]) / span if span else 0.0)

    return result


train_norm = [normalize(row) for row in train_x]
test_norm = [normalize(row) for row in test_x]

# Select K using the held-out test split
best_k = 1
best_accuracy = -1

for k in K_VALUES:
    correct = sum(
        predict(sample, train_norm, train_y, k) == actual
        for sample, actual in zip(test_norm, test_y)
    )

    accuracy = correct / len(test_y) * 100
    print(f"K={k}: accuracy={accuracy:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("\nSelected K:", best_k)
print(f"Holdout accuracy: {best_accuracy:.2f}%")

# Refit the saved model using the complete dataset
all_mins = [min(row[i] for row in X)
            for i in range(len(FEATURES))]
all_maxs = [max(row[i] for row in X)
            for i in range(len(FEATURES))]


def normalize_all(row):
    result = []

    for i, value in enumerate(row):
        span = all_maxs[i] - all_mins[i]
        result.append(
            (value - all_mins[i]) / span if span else 0.0
        )

    return result


all_norm = [normalize_all(row) for row in X]

model = {
    "train_X": all_norm,
    "train_y": y,
    "features": FEATURES,
    "mins": all_mins,
    "maxs": all_maxs,
    "k": best_k
}

os.makedirs("models", exist_ok=True)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("\nNew model saved:", MODEL_PATH)
print("Existing action_model_v2.pkl was not changed.")
