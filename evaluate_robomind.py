
import csv
import os

file_path = os.path.join(
    "outputs",
    "fresh_webcam_test.csv"
)

if not os.path.exists(file_path):
    print("Error: fresh_webcam_test.csv not found!")
    raise SystemExit

with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    columns = reader.fieldnames or []

print("\nRoboMind Model Evaluation")
print("-" * 35)
print("CSV columns:", columns)
print("Total test records:", len(rows))

if not rows:
    print("No test records found.")
    raise SystemExit

# Find the actual and predicted label columns
actual_options = [
    "actual", "actual_label", "actual_action",
    "true_label", "true_action", "ground_truth", "expected"
]

predicted_options = [
    "predicted", "predicted_label", "predicted_action",
    "prediction", "predicted_class"
]

actual_col = next(
    (c for c in columns if c.lower().strip() in actual_options),
    None
)

predicted_col = next(
    (c for c in columns if c.lower().strip() in predicted_options),
    None
)

if actual_col and predicted_col:
    actual = [
        row[actual_col].strip()
        for row in rows
    ]
    predicted = [
        row[predicted_col].strip()
        for row in rows
    ]

    total = len(actual)
    correct = sum(
        a == p for a, p in zip(actual, predicted)
    )

    print(f"\nCorrect predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {correct / total * 100:.2f}%")

    labels = sorted(set(actual) | set(predicted))

    print("\nClass-wise Results")
    print("-" * 35)

    for label in labels:
        tp = sum(
            a == label and p == label
            for a, p in zip(actual, predicted)
        )
        actual_count = sum(a == label for a in actual)
        predicted_count = sum(p == label for p in predicted)

        precision = tp / predicted_count if predicted_count else 0
        recall = tp / actual_count if actual_count else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        print(f"\nAction: {label}")
        print(f"Actual samples: {actual_count}")
        print(f"Correctly predicted: {tp}")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-score: {f1 * 100:.2f}%")

    print("\nConfusion Matrix")
    print("Rows = Actual, Columns = Predicted")
    print("Labels:", labels)
    print("       " + "  ".join(labels))

    for label in labels:
        values = [
            sum(
                a == label and p == predicted_label
                for a, p in zip(actual, predicted)
            )
            for predicted_label in labels
        ]
        print(f"{label}: {values}")

else:
    print("\nActual/predicted label columns were not found.")
    print("Available columns:", columns)
    print(
        "\nSend me the column names printed above, "
        "and I will adapt the evaluation code to your CSV."
    )
