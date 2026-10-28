
import csv
import random
from collections import defaultdict, Counter

input_file = "data/processed/labeled_data.csv"
output_file = "data/processed/balanced_data.csv"

random.seed(42)

# Read the original dataset
with open(input_file, "r", newline="") as file:
    reader = csv.DictReader(file)
    columns = reader.fieldnames
    rows = list(reader)

# Group samples by action
groups = defaultdict(list)

for row in rows:
    groups[row["action"]].append(row)

# Find the smallest class size
minimum = min(len(samples) for samples in groups.values())

# Select equal samples from every class
balanced_rows = []

for action, samples in groups.items():
    balanced_rows.extend(
        random.sample(samples, minimum)
    )

# Shuffle the balanced dataset
random.shuffle(balanced_rows)

# Save to a new file
with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(balanced_rows)

print("Original samples:", len(rows))
print("Balanced samples:", len(balanced_rows))
print("\nBalanced action counts:")

for action, count in Counter(
    row["action"] for row in balanced_rows
).items():
    print(action, ":", count)

print("\nSaved to:", output_file)
