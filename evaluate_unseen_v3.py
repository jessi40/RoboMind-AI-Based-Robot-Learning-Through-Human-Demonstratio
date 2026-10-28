
import csv
import pickle
import numpy as np
from collections import Counter

TEST_FILE = "data/processed/test_data_v3.csv"
MODEL_FILE = "models/action_model_v3.pkl"

FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

LABELS = ["bend_arm", "lower_hand", "raise_hand"]

# Load unseen test data
X_test = []
y_test = []

with open(TEST_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        X_test.append([float(row[name]) for name in FEATURES])
        y_test.append(row["action"])

X_test = np.array(X_test, dtype=float)

# Load V3 model
with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

train_X = np.asarray(model["train_X"], dtype=float)
train_y = np.asarray(model["train_y"])
mins = np.asarray(model["mins"], dtype=float)
maxs = np.asarray(model["maxs"], dtype=float)
k = int(model["k"])

# Normalize test data using training limits
ranges = maxs - mins
ranges[ranges == 0] = 1

X_test = (X_test - mins) / ranges

# Predict using KNN
predictions = []

for sample in X_test:
    distances = np.sqrt(np.sum((train_X - sample) ** 2, axis=1))
    nearest = np.argsort(distances)[:k]
    votes = Counter(train_y[nearest])
    prediction = votes.most_common(1)[0][0]
    predictions.append(prediction)

# Calculate accuracy
correct = sum(
    actual == predicted
    for actual, predicted in zip(y_test, predictions)
)

accuracy = correct / len(y_test) if y_test else 0

print("\nROBOMIND V3 - UNSEEN DATA EVALUATION")
print("=" * 45)
print("Test samples:", len(y_test))
print("K value:", k)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate metrics without scikit-learn
print("\nClassification Report")
print("-" * 55)
print(f"{'Action':<15}{'Precision':>12}{'Recall':>10}{'F1-score':>12}")

for label in LABELS:
    tp = sum(
        actual == label and predicted == label
        for actual, predicted in zip(y_test, predictions)
    )

    fp = sum(
        actual != label and predicted == label
        for actual, predicted in zip(y_test, predictions)
    )

    fn = sum(
        actual == label and predicted != label
        for actual, predicted in zip(y_test, predictions)
    )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(
        f"{label:<15}{precision:>12.3f}"
        f"{recall:>10.3f}{f1:>12.3f}"
    )

# Confusion matrix
print("\nConfusion Matrix")
print("Rows = actual, columns = predicted")
print(f"{'Actual':<15}" + "".join(f"{label:>15}" for label in LABELS))

for actual_label in LABELS:
    row_values = []

    for predicted_label in LABELS:
        count = sum(
            actual == actual_label and predicted == predicted_label
            for actual, predicted in zip(y_test, predictions)
        )
        row_values.append(count)

    print(f"{actual_label:<15}" + "".join(f"{v:>15}" for v in row_values))

print("\nTest samples per action:")
for label in LABELS:
    print(f"{label}: {y_test.count(label)}")

