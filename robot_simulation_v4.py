import cv2
import mediapipe as mp
import math
import pickle
import csv
import os
import time
import numpy as np
from collections import Counter, deque
from datetime import datetime

# =====================================================
# 1. FILE PATHS AND ACTION LOGGING
# =====================================================
MODEL_FILE = "models/action_model_v4.pkl"
LOG_FILE = "outputs/robot_action_log_v4.csv"
os.makedirs("outputs", exist_ok=True)


def log_action(action):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(LOG_FILE) == 0:
            writer.writerow(["timestamp", "action"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action])


# =====================================================
# 2. LOAD V4 MODEL
# =====================================================
with open(MODEL_FILE, "rb") as file:
    saved = pickle.load(file)

model = saved["model"]
features = saved["features"]
mins = np.asarray(saved["mins"], dtype=float)
maxs = np.asarray(saved["maxs"], dtype=float)

print("=" * 45)
print("       ROBOMIND V4 SIMULATION")
print("=" * 45)
print("Model:", MODEL_FILE)
print("Features:", features)

# =====================================================
# 3. MEDIAPIPE SETUP
# =====================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# =====================================================
# 4. CALCULATE ANGLE
# =====================================================
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    lengths = math.hypot(*ba) * math.hypot(*bc)

    if lengths == 0:
        return None

    cosine = max(-1.0, min(1.0, dot / lengths))
    return round(math.degrees(math.acos(cosine)), 2)


# =====================================================
# 5. EXTRACT V4 FEATURES
# =====================================================
def extract_features(landmarks):
    def point(index):
        return (landmarks[index].x, landmarks[index].y)

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


# =====================================================
# 6. NORMALIZE FEATURES AND PREDICT
# =====================================================
def normalize(values):
    values = np.asarray(values, dtype=float)
    spans = maxs - mins
    safe_spans = np.where(spans == 0, 1.0, spans)
    normalized = (values - mins) / safe_spans
    return normalized.reshape(1, -1)


def predict_action(values):
    return str(model.predict(normalize(values))[0])


# =====================================================
# 7. ROBOT ARM ANIMATION
# =====================================================
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
    speed = 0.12
    current_upper += (target_upper - current_upper) * speed
    current_elbow += (target_elbow - current_elbow) * speed

    cv2.putText(canvas, "ROBOMIND V4", (100, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)

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

    cv2.rectangle(canvas, (195, 350), (305, 390), (120, 120, 120), -1)
    cv2.rectangle(canvas, (215, 390), (285, 410), (80, 80, 80), -1)
    cv2.line(canvas, shoulder, elbow, (220, 100, 40), 18)
    cv2.line(canvas, elbow, hand, (40, 160, 220), 14)
    cv2.circle(canvas, shoulder, 20, (60, 60, 60), -1)
    cv2.circle(canvas, elbow, 17, (40, 180, 240), -1)
    cv2.circle(canvas, hand, 13, (40, 180, 80), -1)
    cv2.putText(canvas, "Action: " + action, (35, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    return canvas


# =====================================================
# 8. SESSION STATE
# =====================================================
session_counts = Counter()
prediction_history = deque(maxlen=7)
tracking_started = False
tracking_start_time = None
action = "Waiting"
display_action = "Press R to Start"
previous_logged_action = None

# =====================================================
# 9. START WEBCAM
# =====================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam. Close other camera apps and retry.")

print("\nControls:")
print("Press R to start tracking and action counting.")
print("Press Q to quit.")

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

            # Mirror the webcam image for a more natural preview.
            frame = cv2.flip(frame, 1)

            # IMPORTANT: no recognition, counting, or logging until R is pressed.
            if tracking_started:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                    values = extract_features(result.pose_landmarks.landmark)

                    if all(value is not None and math.isfinite(value) for value in values):
                        predicted = predict_action(values)
                        prediction_history.append(predicted)
                        votes = Counter(prediction_history)
                        stable_action, count = votes.most_common(1)[0]

                        # Require at least four matching predictions for stability.
                        if count >= 4 and stable_action != action:
                            action = stable_action
                            if action != "Waiting" and action != previous_logged_action:
                                log_action(action)
                                previous_logged_action = action
                                session_counts[action] += 1
                                print("V4 action logged:", action)

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
            else:
                # Show the camera preview only; do not process poses or count actions.
                display_action = "Not started"
                action = "Waiting"
                cv2.putText(
                    frame, "CAMERA READY - PRESS R TO START", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )

            # Duration starts only after tracking begins.
            elapsed = int(time.time() - tracking_start_time) if tracking_start_time else 0
            minutes, seconds = divmod(elapsed, 60)

            cv2.putText(frame, "Detected: " + display_action, (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Robot: " + action, (15, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "R = Start Tracking | Q = Quit", (15, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            robot = draw_robot(action)

            # Session dashboard
            dashboard = np.full((500, 300, 3), 250, dtype=np.uint8)
            cv2.putText(dashboard, "V4 SESSION DASHBOARD", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
            cv2.putText(dashboard, f"Status: {'TRACKING' if tracking_started else 'READY'}",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 130, 0) if tracking_started else (0, 100, 180), 2)
            cv2.putText(dashboard, f"Duration: {minutes:02d}:{seconds:02d}",
                        (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
            cv2.putText(dashboard, "Action counts:", (15, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

            y = 195
            for name in ["raise_hand", "lower_hand", "bend_arm"]:
                cv2.putText(dashboard, f"{name}: {session_counts[name]}",
                            (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2)
                y += 40

            total = sum(session_counts.values())
            cv2.putText(dashboard, f"Total actions: {total}", (15, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
            cv2.putText(dashboard, "Model: V4 (71.76% test accuracy)", (10, 385),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1)
            cv2.putText(dashboard, "Log: robot_action_log_v4.csv", (10, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1)

            frame = cv2.resize(frame, (640, 500))
            combined = np.hstack([frame, robot, dashboard])
            cv2.imshow("RoboMind V4 Simulation", combined)

            # Read the key once per loop. Press R to begin tracking.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and not tracking_started:
                tracking_started = True
                tracking_start_time = time.time()
                prediction_history.clear()
                previous_logged_action = None
                action = "Waiting"
                display_action = "Tracking started"
                print("Tracking started. Move your arms to perform actions.")
            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

    elapsed = int(time.time() - tracking_start_time) if tracking_start_time else 0
    print("\n" + "=" * 40)
    print("ROBOMIND V4 SESSION SUMMARY")
    print("=" * 40)
    print("Tracking duration:", elapsed, "seconds")
    for name in ["raise_hand", "lower_hand", "bend_arm"]:
        print(f"{name}: {session_counts[name]}")
    print("Total logged actions:", sum(session_counts.values()))
    print("Log file:", LOG_FILE)
    print("Simulation closed.")
