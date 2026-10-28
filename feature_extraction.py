
import csv
import math
import os

input_file = "data/raw/demonstration.csv"
output_file = "data/processed/features.csv"

os.makedirs("data/processed", exist_ok=True)


def calculate_angle(a, b, c):
    # Calculate angle at point B
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    len_ba = math.hypot(ba[0], ba[1])
    len_bc = math.hypot(bc[0], bc[1])

    if len_ba == 0 or len_bc == 0:
        return ""

    cosine = dot / (len_ba * len_bc)
    cosine = max(-1, min(1, cosine))

    return round(math.degrees(math.acos(cosine)), 2)


with open(input_file, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

if not rows:
    print("No demonstration data found.")
    raise SystemExit

features = []

for row in rows:
    # Left arm: shoulder, elbow, wrist
    ls = (float(row["x11"]), float(row["y11"]))
    le = (float(row["x13"]), float(row["y13"]))
    lw = (float(row["x15"]), float(row["y15"]))

    # Right arm: shoulder, elbow, wrist
    rs = (float(row["x12"]), float(row["y12"]))
    re = (float(row["x14"]), float(row["y14"]))
    rw = (float(row["x16"]), float(row["y16"]))

    # Hip positions
    lh = (float(row["x23"]), float(row["y23"]))
    rh = (float(row["x24"]), float(row["y24"]))

    features.append({
        "frame": row["frame"],
        "time": row["time"],
        "left_elbow_angle": calculate_angle(ls, le, lw),
        "right_elbow_angle": calculate_angle(rs, re, rw),
        "left_shoulder_angle": calculate_angle(lh, ls, le),
        "right_shoulder_angle": calculate_angle(rh, rs, re)
    })

with open(output_file, "w", newline="") as file:
    columns = list(features[0].keys())
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(features)

print("Feature extraction completed!")
print("Frames processed:", len(features))
print("Saved to:", output_file)
print("\nFirst 5 rows:")

for row in features[:5]:
    print(row)
