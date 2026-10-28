import cv2
import mediapipe as mp
import math
import pickle
import csv
import os
import time
from datetime import datetime
from collections import Counter, deque

import numpy as np

# ==================================================
# 1. FILE PATHS AND SETTINGS
# ==================================================

MODEL_FILE = "models/action_model_v2.pkl"
LOG_FILE = "outputs/robot_action_log.csv"
TEST_FILE = "outputs/fresh_webcam_test.csv"

os.makedirs("outputs", exist_ok=True)

# ==================================================
# 2. LOAD TRAINED KNN MODEL
# ==================================================

with open(MODEL_FILE, "rb") as file:
    saved = pickle.load(file)

train_X = saved["train_X"]
train_y = saved["train_y"]
mins = saved["mins"]
maxs = saved["maxs"]
k = saved["k"]

print("=" * 50)
print("        ROBOMIND ROBOT SIMULATION")
print("=" * 50)
print("Model:", MODEL_FILE)
print("K value:", k)

# ==================================================
# 3. MEDIAPIPE SETUP
# ==================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ==================================================
# 4. ACTION LOGGER CONNECTION
# ==================================================

# Use the separate action_logger.py file.
# This keeps all action logging in one place.
from action_logger import log_action

# ==================================================
# 5. CALCULATE BODY ANGLES
# ==================================================

def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    lengths = math.hypot(*ba) * math.hypot(*bc)

    if lengths == 0:
        return None

    cosine = max(-1.0, min(1.0, dot / lengths))
    return round(math.degrees(math.acos(cosine)), 2)


# ==================================================
# 6. EXTRACT FEATURES
# ==================================================

def extract_features(landmarks):
    def point(index):
        p = landmarks[index]
        return (p.x, p.y)

    left_shoulder = point(11)
    left_elbow = point(13)
    left_wrist = point(15)
    left_hip = point(23)

    right_shoulder = point(12)
    right_elbow = point(14)
    right_wrist = point(16)
    right_hip = point(24)

    return [
        calculate_angle(left_shoulder, left_elbow, left_wrist),
        calculate_angle(right_shoulder, right_elbow, right_wrist),
        calculate_angle(left_hip, left_shoulder, left_elbow),
        calculate_angle(right_hip, right_shoulder, right_elbow),
    ]


# ==================================================
# 7. NORMALIZE FEATURES
# ==================================================

def normalize(values):
    result = []

    for i, value in enumerate(values):
        span = maxs[i] - mins[i]

        if span == 0:
            result.append(0.0)
        else:
            result.append((value - mins[i]) / span)

    return result


# ==================================================
# 8. KNN ACTION PREDICTION
# ==================================================

def predict_action(values):
    sample = normalize(values)
    distances = []

    for train_sample, label in zip(train_X, train_y):
        distance = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(sample, train_sample))
        )
        distances.append((distance, label))

    nearest = sorted(distances, key=lambda item: item[0])[:min(k, len(distances))]
    votes = Counter(label for _, label in nearest)

    # Use nearest distance to resolve voting ties.
    return max(
        votes,
        key=lambda label: (
            votes[label],
            -min(distance for distance, name in nearest if name == label),
        ),
    )


# ==================================================
# 9. ROBOT ARM ANIMATION
# ==================================================

current_upper = 60.0
current_elbow = 25.0

def draw_robot(action):
    global current_upper, current_elbow

    canvas = np.full((500, 500, 3), 245, dtype=np.uint8)

    targets = {
        "raise_hand": (110.0, 20.0),
        "lower_hand": (35.0, 15.0),
        "bend_arm": (75.0, 85.0),
        "Waiting": (60.0, 25.0),
    }

    target_upper, target_elbow = targets.get(action, targets["Waiting"])

    # Smooth movement.
    speed = 0.12
    current_upper += (target_upper - current_upper) * speed
    current_elbow += (target_elbow - current_elbow) * speed

    cv2.putText(
        canvas, "ROBOMIND 2D ROBOT", (60, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2
    )

    shoulder = (250, 340)
    upper_length = 125
    lower_length = 105

    theta = math.radians(current_upper)
    forearm_angle = math.radians(current_upper - current_elbow)

    elbow = (
        int(shoulder[0] + upper_length * math.cos(theta)),
        int(shoulder[1] - upper_length * math.sin(theta)),
    )

    hand = (
        int(elbow[0] + lower_length * math.cos(forearm_angle)),
        int(elbow[1] - lower_length * math.sin(forearm_angle)),
    )

    # Robot base.
    cv2.rectangle(canvas, (195, 350), (305, 390), (120, 120, 120), -1)
    cv2.rectangle(canvas, (215, 390), (285, 410), (80, 80, 80), -1)

    # Robot arms.
    cv2.line(canvas, shoulder, elbow, (220, 100, 40), 18)
    cv2.line(canvas, elbow, hand, (40, 160, 220), 14)

    # Joints and hand.
    cv2.circle(canvas, shoulder, 20, (60, 60, 60), -1)
    cv2.circle(canvas, elbow, 17, (40, 180, 240), -1)
    cv2.circle(canvas, hand, 13, (40, 180, 80), -1)

    cv2.putText(
        canvas, "Action: " + action, (35, 455),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2
    )

    return canvas


# ==================================================
# 10. LOAD PREVIOUS WEBCAM TEST ACCURACY
# ==================================================

def get_test_accuracy():
    if not os.path.exists(TEST_FILE):
        return None

    try:
        with open(TEST_FILE, "r", newline="") as file:
            rows = list(csv.DictReader(file))

        if not rows:
            return None

        correct = sum(
            row.get("correct", "0") in ("1", "True")
            for row in rows
        )
        return 100.0 * correct / len(rows)

    except (OSError, csv.Error, ValueError):
        return None


test_accuracy = get_test_accuracy()

# ==================================================
# 11. SESSION STATISTICS
# ==================================================

session_counts = Counter()
session_start = time.time()
prediction_history = deque(maxlen=7)

action = "Waiting"
display_action = "No person detected"
previous_logged_action = None

# ==================================================
# 12. START WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Cannot open webcam. Close other camera apps and retry.")

print("\nControls:")
print("Show your body clearly to the webcam.")
print("Perform raise_hand, lower_hand, and bend_arm.")
print("Press Q to quit the simulation.")

try:
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            success, frame = cap.read()

            if not success:
                print("Cannot read webcam frame.")
                break

            # Keep the same orientation used during collection.
            # Do not flip before MediaPipe pose detection.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                values = extract_features(result.pose_landmarks.landmark)

                if all(
                    value is not None and math.isfinite(value)
                    for value in values
                ):
                    predicted = predict_action(values)
                    prediction_history.append(predicted)
                    votes = Counter(prediction_history)
                    stable_action, count = votes.most_common(1)[0]

                    # Require at least 4 matching predictions before switching.
                    if count >= 4 and stable_action != action:
                        action = stable_action

                        if action != "Waiting":
                            # Save action to outputs/robot_action_log.csv
                            # using action_logger.py, only when it changes.
                            if action != previous_logged_action:
                                log_action(action)
                                previous_logged_action = action
                                session_counts[action] += 1
                                print("Robot action logged:", action)

                    display_action = predicted

                else:
                    prediction_history.clear()
                    action = "Waiting"
                    display_action = "Pose unclear"
                    previous_logged_action = None

            else:
                prediction_history.clear()
                action = "Waiting"
                display_action = "No person detected"
                previous_logged_action = None

            # Session duration.
            elapsed = int(time.time() - session_start)
            minutes = elapsed // 60
            seconds = elapsed % 60

            # Webcam status.
            cv2.putText(
                frame, "Detected: " + display_action, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )
            cv2.putText(
                frame, "Robot: " + action, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
            )
            cv2.putText(
                frame, "Press Q to quit", (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            # Robot panel.
            robot = draw_robot(action)

            # Live session dashboard.
            dashboard = np.full((500, 300, 3), 250, dtype=np.uint8)
            cv2.putText(
                dashboard, "SESSION DASHBOARD", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2
            )
            cv2.putText(
                dashboard, f"Duration: {minutes:02d}:{seconds:02d}", (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2
            )
            cv2.putText(
                dashboard, "Action counts:", (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2
            )

            y = 180
            for name in ["raise_hand", "lower_hand", "bend_arm"]:
                cv2.putText(
                    dashboard, f"{name}: {session_counts[name]}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2
                )
                y += 45

            total_actions = sum(session_counts.values())
            cv2.putText(
                dashboard, f"Total actions: {total_actions}", (15, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2
            )

            if test_accuracy is not None:
                accuracy_text = f"Past test accuracy: {test_accuracy:.2f}%"
            else:
                accuracy_text = "Past test accuracy: N/A"

            cv2.putText(
                dashboard, accuracy_text, (15, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 2
            )
            cv2.putText(
                dashboard, "Log: outputs/robot_action_log.csv", (15, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1
            )

            # Resize and combine all three panels.
            frame = cv2.resize(frame, (640, 500))
            combined = np.hstack([frame, robot, dashboard])

            cv2.imshow("RoboMind - Intelligent Robot Simulation", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

    elapsed = int(time.time() - session_start)
    print("\n" + "=" * 40)
    print("ROBOMIND SESSION SUMMARY")
    print("=" * 40)
    print("Session duration:", elapsed, "seconds")

    for name in ["raise_hand", "lower_hand", "bend_arm"]:
        print(f"{name}: {session_counts[name]}")

    print("Total logged actions:", sum(session_counts.values()))
    print("Action log:", LOG_FILE)
    print("Simulation closed.")
