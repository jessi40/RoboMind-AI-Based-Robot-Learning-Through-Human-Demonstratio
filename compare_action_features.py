
import csv
from collections import defaultdict

FILES = {
    "Training": "data/processed/balanced_data.csv",
    "Fresh Webcam": "outputs/fresh_webcam_test.csv",
}

FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
]

LABEL_COLUMNS = {
    "Training": "action",
    "Fresh Webcam": "actual_action",
}

def load_data(path, label_column):
    groups = defaultdict(list)

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            label = row[label_column].strip()
            values = {
                feature: float(row[feature])
                for feature in FEATURES
            }
            groups[label].append(values)

    return groups

training = load_data(
    FILES["Training"], LABEL_COLUMNS["Training"]
)
webcam = load_data(
    FILES["Fresh Webcam"], LABEL_COLUMNS["Fresh Webcam"]
)

print("ACTION-BY-ACTION FEATURE COMPARISON")

for action in ["raise_hand", "lower_hand", "bend_arm"]:
    print("\n" + "=" * 55)
    print("Action:", action)

    for feature in FEATURES:
        train_values = [
            row[feature] for row in training.get(action, [])
        ]
        test_values = [
            row[feature] for row in webcam.get(action, [])
        ]

        if not train_values or not test_values:
            print(feature, "- data unavailable")
            continue

        train_mean = sum(train_values) / len(train_values)
        test_mean = sum(test_values) / len(test_values)

        print(f"\n{feature}")
        print(f"  Training mean: {train_mean:.2f}")
        print(f"  Webcam mean:   {test_mean:.2f}")
        print(f"  Difference:    {test_mean - train_mean:+.2f}")

print("\nComparison complete. No files were changed.")
