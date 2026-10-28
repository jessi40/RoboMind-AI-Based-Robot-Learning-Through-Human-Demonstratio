
import pandas as pd

# Load the V4 training dataset
df = pd.read_csv("data/processed/labeled_data_v4.csv")

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle"
]

# Display angle statistics for each action
print("===== ACTION ANGLE DISTRIBUTIONS =====")

for action, group in df.groupby("action"):
    print(f"\nAction: {action}")
    print("Number of frames:", len(group))

    print(
        group[features]
        .agg(["mean", "std", "min", "median", "max"])
        .round(2)
    )

# Compare lower_hand and raise_hand
print("\n===== LOWER HAND VS RAISE HAND =====")

for feature in features:
    lower = df.loc[df["action"] == "lower_hand", feature]
    raised = df.loc[df["action"] == "raise_hand", feature]

    print(f"\n{feature}")
    print("Lower hand mean:", round(lower.mean(), 2))
    print("Raise hand mean:", round(raised.mean(), 2))
    print(
        "Lower hand range:",
        round(lower.min(), 2), "-",
        round(lower.max(), 2)
    )
    print(
        "Raise hand range:",
        round(raised.min(), 2), "-",
        round(raised.max(), 2)
    )
