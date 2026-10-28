
import csv
from collections import defaultdict

files = {
    "TRAIN": "data/processed/balanced_data_new.csv",
    "TEST": "data/processed/test_data_v3.csv"
}

actions = ["raise_hand", "lower_hand", "bend_arm"]

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

data = defaultdict(list)

for dataset, path in files.items():
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["action"] in actions:
                values = [
                    float(row[name])
                    for name in features
                ]
                data[(dataset, row["action"])].append(values)

for action in actions:
    print("\n" + "=" * 60)
    print("ACTION:", action)

    for dataset in ["TRAIN", "TEST"]:
        samples = data[(dataset, action)]

        print(f"\n{dataset} - first 5 samples")

        for sample in samples[:5]:
            print(
                "Left elbow:", round(sample[0], 2),
                "| Right elbow:", round(sample[1], 2),
                "| Left shoulder:", round(sample[2], 2),
                "| Right shoulder:", round(sample[3], 2)
            )
