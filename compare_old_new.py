
import csv
import os
from collections import defaultdict

files = {
    "OLD": "data/processed/labeled_data.csv",
    "NEW": "data/processed/labeled_data_new.csv"
}

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

for name, path in files.items():
    print("\n" + "=" * 50)
    print(name, "DATASET")
    print("=" * 50)

    if not os.path.exists(path):
        print("File not found:", path)
        continue

    grouped = defaultdict(list)

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            try:
                values = [float(row[f]) for f in features]
                grouped[row["action"]].append(values)
            except (ValueError, KeyError):
                continue

    for action in ["raise_hand", "lower_hand", "bend_arm"]:
        rows = grouped[action]
        print(f"\n{action}: {len(rows)} samples")

        for i, feature in enumerate(features):
            values = [row[i] for row in rows]

            if values:
                mean = sum(values) / len(values)
                print(
                    f"{feature}: "
                    f"min={min(values):.1f}, "
                    f"max={max(values):.1f}, "
                    f"mean={mean:.1f}"
                )
