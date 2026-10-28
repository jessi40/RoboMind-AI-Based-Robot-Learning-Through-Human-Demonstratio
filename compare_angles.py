
import csv
from collections import defaultdict

FILES = {
    "Training": "data/processed/labeled_data.csv",
    "Webcam": "outputs/fresh_webcam_test.csv"
}

FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

for source, filename in FILES.items():
    groups = defaultdict(list)

    with open(filename, "r", newline="") as file:
        rows = csv.DictReader(file)

        for row in rows:
            action = (
                row["action"]
                if source == "Training"
                else row["actual_action"]
            )

            for feature in FEATURES:
                groups[(action, feature)].append(
                    float(row[feature])
                )

    print(f"\n{'=' * 15} {source} DATA {'=' * 15}")

    for action in ["raise_hand", "lower_hand", "bend_arm"]:
        print(f"\n{action}:")

        for feature in FEATURES:
            values = groups[(action, feature)]

            if values:
                average = sum(values) / len(values)
                print(f"  {feature}: {average:.2f} degrees")
            else:
                print(f"  {feature}: No data")
