
import cv2
import mediapipe as mp
import math
import pickle
import csv
import os
from collections import Counter

# 1. File paths
MODEL_FILE = "models/action_model_v2.pkl"
OUTPUT_FILE = "outputs/fresh_webcam_test.csv"

# 2. Load the trained model
with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

print("RoboMind Fresh Webcam Test")
print("Model:", MODEL_FILE)
print("K value:", model["k"])

# 3. Set up MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

ACTIONS = {
    ord("1"): "raise_hand",
    ord("2"): "lower_hand",
    ord("3"): "bend_arm"
}

ACTION_NAMES = ["raise_hand", "lower_hand", "bend_arm"]

expected_action = None
recording = False
test_rows = []

# 4. Calculate angle between three points
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    lengths = math.hypot(*ba) * math.hypot(*bc)

    if lengths == 0:
        return None

    cosine = max(-1.0, min(1.0, dot / lengths))

    return round(math.degrees(math.acos(cosine)), 2)


# 5. Extract features in training-data order
def extract_features(landmarks):
    def point(index):
        p = landmarks[index]
        return (p.x, p.y)

    # Left arm landmarks
    left_shoulder = point(11)
    left_elbow = point(13)
    left_wrist = point(15)
    left_hip = point(23)

    # Right arm landmarks
    right_shoulder = point(12)
    right_elbow = point(14)
    right_wrist = point(16)
    right_hip = point(24)

    # Elbow angles
    left_elbow_angle = calculate_angle(
        left_shoulder, left_elbow, left_wrist
    )

    right_elbow_angle = calculate_angle(
        right_shoulder, right_elbow, right_wrist
    )

    # Shoulder angles
    left_shoulder_angle = calculate_angle(
        left_hip, left_shoulder, left_elbow
    )

    right_shoulder_angle = calculate_angle(
        right_hip, right_shoulder, right_elbow
    )

    return [
        left_elbow_angle,
        right_elbow_angle,
        left_shoulder_angle,
        right_shoulder_angle
    ]


# 6. Normalize using training-model ranges
def normalize(values, mins, maxs):
    result = []

    for i, value in enumerate(values):
        span = maxs[i] - mins[i]

        if span == 0:
            result.append(0.0)
        else:
            result.append((value - mins[i]) / span)

    return result


# 7. Predict action using KNN
def predict_action(values, model):
    mins = model["mins"]
    maxs = model["maxs"]
    train_X = model["train_X"]
    train_y = model["train_y"]
    k = model["k"]

    sample = normalize(values, mins, maxs)

    distances = []

    for train_sample, label in zip(train_X, train_y):
        distance = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(sample, train_sample)
            )
        )

        distances.append((distance, label))

    nearest = sorted(
        distances,
        key=lambda item: item[0]
    )[:min(k, len(distances))]

    votes = Counter(label for _, label in nearest)

    # Resolve voting ties using the nearest neighbor
    return max(
        votes,
        key=lambda label: (
            votes[label],
            -min(
                d for d, name in nearest
                if name == label
            )
        )
    )


# 8. Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit(
        "Cannot open webcam. Close other camera apps and try again."
    )

try:
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            success, frame = cap.read()

            if not success:
                print("Cannot read webcam frame.")
                break

            # IMPORTANT:
            # Do not flip the frame before pose detection.
            # Training data was collected without flipping.

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = pose.process(rgb)

            predicted_action = "No person detected"
            values = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                mp_draw.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                values = extract_features(landmarks)

                # Check that all four angles are valid
                if all(
                    v is not None and math.isfinite(v)
                    for v in values
                ):
                    predicted_action = predict_action(
                        values, model
                    )

                    if recording and expected_action is not None:
                        test_rows.append([
                            expected_action,
                            predicted_action,
                            *values,
                            int(
                                expected_action
                                == predicted_action
                            )
                        ])
                else:
                    predicted_action = "Invalid pose angles"

            # 9. Display status
            status = (
                "RECORDING"
                if recording
                else "NOT RECORDING"
            )

            actual = (
                expected_action
                if expected_action
                else "Select action: 1, 2, 3"
            )

            cv2.putText(
                frame,
                "Actual: " + actual,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Predicted: " + predicted_action,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                status,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255) if recording
                else (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "1 Raise | 2 Lower | 3 Bend",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "R: Start/Stop recording | Q: Finish",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Samples: " + str(len(test_rows)),
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "RoboMind - Fresh Webcam Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Select the actual action
            if key in ACTIONS:
                expected_action = ACTIONS[key]
                print(
                    "Actual action selected:",
                    expected_action
                )

            # Start or stop recording
            elif key == ord("r"):
                if expected_action is None:
                    print(
                        "First select an action using 1, 2, or 3."
                    )
                else:
                    recording = not recording
                    print("Recording:", recording)

            # Finish testing
            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()


# 10. Save and evaluate test results
if test_rows:
    os.makedirs("outputs", exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "actual_action",
            "predicted_action",
            "left_elbow_angle",
            "right_elbow_angle",
            "left_shoulder_angle",
            "right_shoulder_angle",
            "correct"
        ])

        writer.writerows(test_rows)

    correct = sum(row[-1] for row in test_rows)
    total = len(test_rows)
    accuracy = 100 * correct / total

    print("\n--- FRESH WEBCAM TEST RESULTS ---")
    print("Total samples:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", total - correct)
    print(f"Overall accuracy: {accuracy:.2f}%")

    print("\nPer-action accuracy:")

    for action in ACTION_NAMES:
        action_rows = [
            row for row in test_rows
            if row[0] == action
        ]

        if action_rows:
            action_correct = sum(
                row[-1] for row in action_rows
            )

            action_accuracy = (
                100 * action_correct / len(action_rows)
            )

            print(
                f"{action}: "
                f"{action_correct}/{len(action_rows)} "
                f"({action_accuracy:.2f}%)"
            )

    print("\nPredicted action counts:")
    predictions = Counter(
        row[1] for row in test_rows
    )

    for action, count in predictions.items():
        print(f"{action}: {count}")

    print("\nResults saved to:", OUTPUT_FILE)

else:
    print("No test samples recorded. Run the test again.")