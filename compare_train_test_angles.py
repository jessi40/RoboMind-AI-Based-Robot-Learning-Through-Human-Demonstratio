
import csv
from collections import defaultdict

FILES = {
    "TRAIN": "data/processed/balanced_data_new.csv",
    "TEST": "data/processed/test_data_v3.csv"
}

FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

data = defaultdict(list)

for dataset_name, file_path in FILES.items():
    with open(file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            action = row["action"]

            for feature in FEATURES:
                data[(dataset_name, action, feature)].append(
                    float(row[feature])
                )

print("TRAINING vs TEST ANGLE COMPARISON")
print("=" * 75)

for action in ["raise_hand", "lower_hand", "bend_arm"]:
    print(f"\nACTION: {action}")

    for feature in FEATURES:
        print(f"\n{feature}")

        for dataset_name in ["TRAIN", "TEST"]:
            values = data[(dataset_name, action, feature)]

            mean_value = sum(values) / len(values)

            print(
                f"  {dataset_name}: "
                f"count={len(values)}, "
                f"min={min(values):.2f}, "
                f"max={max(values):.2f}, "
                f"mean={mean_value:.2f}"
            )
