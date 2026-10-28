
import csv
from collections import defaultdict

PATH = "data/processed/labeled_data_new.csv"

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

groups = defaultdict(list)

with open(PATH, "r", newline="", encoding="utf-8-sig") as file:
    for row in csv.DictReader(file):
        groups[row["action"]].append(row)

for action, rows in groups.items():
    print("\n" + "=" * 45)
    print(f"{action}: {len(rows)} samples")

    for feature in features:
        values = [float(row[feature]) for row in rows]
        values.sort()

        n = len(values)
        median = values[n // 2]

        low_index = int(n * 0.10)
        high_index = min(int(n * 0.90), n - 1)

        print(
            f"{feature}: "
            f"10th percentile approx={values[low_index]:.1f}, "
            f"median={median:.1f}, "
            f"90th percentile approx={values[high_index]:.1f}"
        )
