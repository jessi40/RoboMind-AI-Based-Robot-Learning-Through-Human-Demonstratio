
import csv
import os
from collections import Counter

PATH = "data/processed/labeled_data_new.csv"

if not os.path.exists(PATH):
    print("Dataset file not found:", PATH)
    raise SystemExit

with open(PATH, "r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

print("ROBOMIND NEW DATASET CHECK")
print("=" * 40)
print("Total samples:", len(rows))

required_columns = [
    "time",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "action"
]

if not rows:
    print("Dataset is empty.")
    raise SystemExit

print("\nColumns:", list(rows[0].keys()))

missing_columns = [
    column for column in required_columns
    if column not in rows[0]
]

if missing_columns:
    print("Missing columns:", missing_columns)
    raise SystemExit

counts = Counter(row["action"] for row in rows)

print("\nSamples per action:")
for action in ["raise_hand", "lower_hand", "bend_arm"]:
    print(f"{action}: {counts[action]}")

features = required_columns[1:5]
invalid_count = 0

for row_number, row in enumerate(rows, start=2):
    for feature in features:
        try:
            value = float(row[feature])
            if not 0 <= value <= 180:
                invalid_count += 1
        except (ValueError, TypeError):
            invalid_count += 1

print("\nInvalid angle values:", invalid_count)

print("\nAngle ranges:")
for action in ["raise_hand", "lower_hand", "bend_arm"]:
    action_rows = [r for r in rows if r["action"] == action]

    print(f"\n{action}")

    for feature in features:
        values = []

        for row in action_rows:
            try:
                values.append(float(row[feature]))
            except (ValueError, TypeError):
                pass

        if values:
            print(
                f"{feature}: "
                f"min={min(values):.1f}, "
                f"max={max(values):.1f}, "
                f"mean={sum(values)/len(values):.1f}"
            )

print("\nDataset check completed.")
