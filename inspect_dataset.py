
import csv
import os
from collections import Counter

FILES = {
    "Original dataset": "data/processed/labeled_data.csv",
    "Balanced dataset": "data/processed/balanced_data.csv",
    "Fresh webcam test": "outputs/fresh_webcam_test.csv",
}

FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
]

def inspect_file(title, path):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if not os.path.exists(path):
        print("File not found:", path)
        return

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    print("File:", path)
    print("Rows:", len(rows))
    print("Columns:", columns)

    if not rows:
        print("Dataset is empty.")
        return

    # Check action counts
    label_col = next(
        (c for c in columns if c.lower() in
         ["action", "label", "actual_action"]),
        None
    )

    if label_col:
        print("\nAction counts:")
        for label, count in Counter(
            row[label_col] for row in rows
        ).items():
            print(f"  {label}: {count}")

    # Check missing values and feature ranges
    print("\nFeature inspection:")

    for feature in FEATURES:
        if feature not in columns:
            print(f"  {feature}: COLUMN MISSING")
            continue

        values = []
        missing = 0
        invalid = 0

        for row in rows:
            raw = (row.get(feature) or "").strip()

            if not raw:
                missing += 1
                continue

            try:
                value = float(raw)
                if not (-float("inf") < value < float("inf")):
                    invalid += 1
                else:
                    values.append(value)
            except ValueError:
                invalid += 1

        if values:
            print(
                f"  {feature}: "
                f"valid={len(values)}, "
                f"missing={missing}, "
                f"invalid={invalid}, "
                f"min={min(values):.2f}, "
                f"max={max(values):.2f}, "
                f"mean={sum(values) / len(values):.2f}"
            )
        else:
            print(
                f"  {feature}: no valid numeric values; "
                f"missing={missing}, invalid={invalid}"
            )

    # Check recorded test correctness
    if "correct" in columns:
        correct_count = sum(
            (row.get("correct") or "").strip().lower()
            in ["true", "1", "yes"]
            for row in rows
        )
        print(
            f"\nCorrect records: {correct_count}/{len(rows)}"
        )

for title, path in FILES.items():
    inspect_file(title, path)

print("\nInspection complete. No files were changed.")
