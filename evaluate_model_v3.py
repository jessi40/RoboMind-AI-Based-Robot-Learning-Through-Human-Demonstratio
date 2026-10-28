
import csv
from collections import Counter

FILE = "data/processed/balanced_data_new.csv"

FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

# Load dataset
with open(FILE, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Extract features and actual labels
X = [
    [float(row[col]) for col in FEATURES]
    for row in rows
]
y = [row["action"] for row in rows]

# Load V3 model
import pickle

with open("models/action_model_v3.pkl", "rb") as f:
    model = pickle.load(f)

train_X = model["train_X"]
train_y = model["train_y"]
mins = model["mins"]
maxs = model["maxs"]
k = model["k"]

# Normalize a feature row
def normalize(row):
    result = []
    for i, value in enumerate(row):
        if maxs[i] == mins[i]:
            result.append(0.0)
        else:
            result.append(
                (value - mins[i]) / (maxs[i] - mins[i])
            )
    return result

# Predict using the saved KNN model
def predict(row):
    row = normalize(row)

    distances = []
    for i, sample in enumerate(train_X):
        distance = sum(
            (row[j] - sample[j]) ** 2
            for j in range(len(row))
        ) ** 0.5

        distances.append((distance, train_y[i]))

    distances.sort(key=lambda item: item[0])
    nearest = [label for _, label in distances[:k]]

    return Counter(nearest).most_common(1)[0][0]

# Predictions on the dataset
predictions = [predict(row) for row in X]

classes = sorted(set(y) | set(predictions))

correct = sum(
    actual == predicted
    for actual, predicted in zip(y, predictions)
)

print("ROBOMIND V3 EVALUATION")
print("=" * 35)
print("Dataset samples:", len(y))
print("K value:", k)
print("Accuracy: {:.2f}%".format(100 * correct / len(y)))

print("\nClassification Report")
print("-" * 60)
print("{:<15} {:>10} {:>10} {:>10}".format(
    "Action", "Precision", "Recall", "F1-score"
))

for action in classes:
    tp = sum(
        actual == action and predicted == action
        for actual, predicted in zip(y, predictions)
    )
    fp = sum(
        actual != action and predicted == action
        for actual, predicted in zip(y, predictions)
    )
    fn = sum(
        actual == action and predicted != action
        for actual, predicted in zip(y, predictions)
    )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("{:<15} {:>9.2f} {:>9.2f} {:>9.2f}".format(
        action, precision, recall, f1
    ))

# Confusion matrix
print("\nConfusion Matrix")
print("Rows = actual, columns = predicted")
print()

print("{:<15}".format("Actual"), end="")
for action in classes:
    print("{:>15}".format(action), end="")
print()

for actual_action in classes:
    print("{:<15}".format(actual_action), end="")
    for predicted_action in classes:
        count = sum(
            actual == actual_action and predicted == predicted_action
            for actual, predicted in zip(y, predictions)
        )
        print("{:>15}".format(count), end="")
    print()
