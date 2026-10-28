
import csv
from collections import Counter, defaultdict

PATH = "data/processed/labeled_data.csv"

with open(PATH, "r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

print("DATASET RECORDING CHECK")
print("-" * 40)
print("Total rows:", len(rows))

# Count samples per action
counts = Counter(row["action"] for row in rows)

print("\nSamples per action:")
for action, count in counts.items():
    print(f"{action}: {count}")

# Check timestamp ranges per action
times = defaultdict(list)

for row in rows:
    try:
        times[row["action"]].append(float(row["time"]))
    except (ValueError, KeyError):
        pass

print("\nTimestamp ranges:")
for action, values in times.items():
    if values:
        print(
            f"{action}: "
            f"min={min(values):.3f}, "
            f"max={max(values):.3f}, "
            f"unique timestamps={len(set(values))}"
        )

# Count consecutive rows with the same action
runs = []
previous = None

for row in rows:
    action = row["action"]

    if action != previous:
        runs.append(action)
        previous = action

print("\nConsecutive action segments:", len(runs))
print("Action segment sequence:")
print(" -> ".join(runs[:40]))

print("\nCheck complete. No files were changed.")
