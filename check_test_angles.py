
import csv
from collections import defaultdict

file_path = "data/processed/test_data_v3.csv"

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

data = defaultdict(list)

with open(file_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        action = row["action"]
        values = [float(row[name]) for name in features]
        data[action].append(values)

print("TEST DATA ANGLE SUMMARY")
print("=" * 65)

for action, rows in data.items():
    print(f"\nAction: {action}")
    print(f"Samples: {len(rows)}")

    for i, feature in enumerate(features):
        values = [row[i] for row in rows]

        mean_value = sum(values) / len(values)
        min_value = min(values)
        max_value = max(values)

        print(
            f"{feature}: "
            f"min={min_value:.2f}, "
            f"max={max_value:.2f}, "
            f"mean={mean_value:.2f}"
        )

