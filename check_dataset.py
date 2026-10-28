
import csv
from collections import Counter, defaultdict

FILE = "data/processed/labeled_data.csv"

with open(FILE, "r", newline="") as f:
    rows = list(csv.DictReader(f))

print("Total frames:", len(rows))

counts = Counter(row["action"] for row in rows)
print("\nAction counts:")

for action, count in counts.items():
    print(f"{action}: {count}")

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

groups = defaultdict(list)

for row in rows:
    action = row["action"]

    for feature in features:
        groups[(action, feature)].append(
            float(row[feature])
        )

print("\nAverage angle for each action:")

for action in ["raise_hand", "lower_hand", "bend_arm"]:
    print(f"\n{action}:")

    for feature in features:
        values = groups[(action, feature)]

        if values:
            average = sum(values) / len(values)
            print(f"  {feature}: {average:.2f} degrees")

