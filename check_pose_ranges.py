
import csv
from collections import defaultdict

# Dataset file paths
files = {
    "Training": "data/processed/labeled_data.csv",
    "Fresh Webcam": "outputs/fresh_webcam_test.csv"
}

# Features used by the RoboMind model
features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

# Store angle values for each dataset and action
data = defaultdict(lambda: defaultdict(list))

# Read both datasets
for dataset_name, path in files.items():

    with open(path, "r", newline="", encoding="utf-8-sig") as file:

        for row in csv.DictReader(file):

            # Get the action label
            action = row.get("action") or row.get("actual_action")

            if not action:
                continue

            # Store the four angle values
            for feature in features:

                try:
                    angle = float(row[feature])
                    data[(dataset_name, action)][feature].append(angle)

                except (ValueError, KeyError):
                    pass

# Display statistics for every action
print("ROBOMIND POSE RANGE ANALYSIS")
print("=" * 50)

for (dataset_name, action), values in sorted(data.items()):

    print(f"\nDataset: {dataset_name}")
    print(f"Action: {action}")
    print("-" * 50)

    for feature in features:

        numbers = values[feature]

        if numbers:
            minimum = min(numbers)
            maximum = max(numbers)
            mean = sum(numbers) / len(numbers)

            print(
                f"{feature}: "
                f"Min = {minimum:.1f}, "
                f"Max = {maximum:.1f}, "
                f"Mean = {mean:.1f}"
            )
        else:
            print(f"{feature}: No valid data")

print("\nAnalysis completed. No files were modified.")
