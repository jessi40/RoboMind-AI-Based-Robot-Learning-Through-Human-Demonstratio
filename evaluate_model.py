
import csv
import random
import math
from collections import Counter, defaultdict

random.seed(42)

# Load the balanced dataset
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

# Split the data
indices = list(range(len(rows)))
random.shuffle(indices)
split = int(0.8 * len(indices))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = [X[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
X_test = [X[i] for i in test_idx]
y_test = [y[i] for i in test_idx]

# Normalize using training values
mins = [min(r[j] for r in X_train) for j in range(4)]
maxs = [max(r[j] for r in X_train) for j in range(4)]

def normalize(row):
    return [
        (row[j] - mins[j]) / (maxs[j] - mins[j])
        if maxs[j] != mins[j] else 0.0
        for j in range(4)
    ]

X_train = [normalize(r) for r in X_train]
X_test = [normalize(r) for r in X_test]

# KNN prediction with K=1
def predict(sample):
    distances = []

    for train_row, label in zip(X_train, y_train):
        distance = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(sample, train_row))
        )
        distances.append((distance, label))

    return min(distances, key=lambda item: item[0])[1]

labels = sorted(set(y))
matrix = {actual: Counter() for actual in labels}

for actual, sample in zip(y_test, X_test):
    predicted = predict(sample)
    matrix[actual][predicted] += 1

correct = sum(matrix[label][label] for label in labels)
total = len(y_test)

print(f"Overall accuracy: {correct / total * 100:.2f}%")
print("\nPer-action results:")

for label in labels:
    count = sum(matrix[label].values())
    hits = matrix[label][label]
    accuracy = hits / count * 100 if count else 0
    print(f"{label}: {hits}/{count} correct ({accuracy:.2f}%)")

print("\nConfusion matrix")
print("Rows = actual actions; columns = predicted actions")
print("Actual \\ Predicted |", " | ".join(labels))

for actual in labels:
    print(actual, "|", " | ".join(
        str(matrix[actual][predicted]) for predicted in labels
    ))
