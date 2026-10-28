
import pandas as pd

df = pd.read_csv("data/processed/test_data_v4.csv")

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

print("===== TEST DATA ANGLE DISTRIBUTIONS =====")

for action, group in df.groupby("action"):
    print(f"\nAction: {action}")
    print("Number of frames:", len(group))

    print(
        group[features]
        .agg(["mean", "std", "min", "median", "max"])
        .round(2)
    )

print("\n===== RIGHT SHOULDER ANGLE COMPARISON =====")

for action in ["lower_hand", "raise_hand"]:
    values = df.loc[
        df["action"] == action,
        "right_shoulder_angle"
    ]

    print(f"\n{action}")
    print("Mean:", round(values.mean(), 2))
    print("Median:", round(values.median(), 2))
    print("Minimum:", round(values.min(), 2))
    print("Maximum:", round(values.max(), 2))
