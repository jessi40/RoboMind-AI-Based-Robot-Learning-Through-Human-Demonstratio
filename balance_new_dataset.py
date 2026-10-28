
import csv
import os
import random
from collections import Counter, defaultdict

INPUT = "data/processed/labeled_data_new.csv"
OUTPUT = "data/processed/balanced_data_new.csv"

random.seed(42)

with open(INPUT, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    columns = reader.fieldnames
    rows = list(reader)

groups = defaultdict(list)

for row in rows:
    groups[row["action"]].append(row)

actions = ["raise_hand", "lower_hand", "bend_arm"]

if any(not groups[action] for action in actions):
    raise ValueError("One or more actions have no samples.")

target_count = min(len(groups[action]) for action in actions)

balanced_rows = []

for action in actions:
    selected = random.sample(groups[action], target_count)
    balanced_rows.extend(selected)

random.shuffle(balanced_rows)

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(balanced_rows)

print("Balanced dataset saved:", OUTPUT)
print("Total samples:", len(balanced_rows))
print("Samples per action:")

for action, count in sorted(
    Counter(row["action"] for row in balanced_rows).items()
):
    print(f"{action}: {count}")
