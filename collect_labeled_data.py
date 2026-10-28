
import cv2
import mediapipe as mp
import csv
import math
import os
import time

os.makedirs("data/processed", exist_ok=True)

OUTPUT = "data/processed/labeled_data.csv"

# Actions selected using keyboard keys
actions = {
    ord("1"): "raise_hand",
    ord("2"): "lower_hand",
    ord("3"): "bend_arm"
}

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    lengths = math.hypot(*ba) * math.hypot(*bc)

    if lengths == 0:
        return None

    cosine = max(-1, min(1, dot / lengths))
    return round(math.degrees(math.acos(cosine)), 2)


def extract_features(landmarks):
    p = landmarks.landmark

    left_elbow = angle(
        (p[11].x, p[11].y),
        (p[13].x, p[13].y),
        (p[15].x, p[15].y)
    )

    right_elbow = angle(
        (p[12].x, p[12].y),
        (p[14].x, p[14].y),
        (p[16].x, p[16].y)
    )

    left_shoulder = angle(
        (p[23].x, p[23].y),
        (p[11].x, p[11].y),
        (p[13].x, p[13].y)
    )

    right_shoulder = angle(
        (p[24].x, p[24].y),
        (p[12].x, p[12].y),
        (p[14].x, p[14].y)
    )

    return [
        left_elbow,
        right_elbow,
        left_shoulder,
        right_shoulder
    ]


columns = [
    "time",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "action"
]

records = []
current_action = None
recording = False
start_time = 0

cap = cv2.VideoCapture(0)

print("1 = raise_hand")
print("2 = lower_hand")
print("3 = bend_arm")
print("R = start/stop recording")
print("Q = save and quit")

try:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Cannot access webcam")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            if recording and current_action is not None:
                values = extract_features(result.pose_landmarks)

                if all(v is not None for v in values):
                    records.append([
                        round(time.time() - start_time, 3),
                        *values,
                        current_action
                    ])

        label = current_action or "Select action: 1, 2, 3"
        status = "RECORDING" if recording else "READY"

        cv2.putText(
            frame, f"{label} | {status}", (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 0), 2
        )

        cv2.imshow("RoboMind - Labeled Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in actions:
            current_action = actions[key]
            print("Selected:", current_action)

        elif key == ord("r"):
            if current_action is None:
                print("Select an action first using 1, 2, or 3")
            else:
                recording = not recording
                if recording:
                    start_time = time.time()
                    print("Recording:", current_action)
                else:
                    print("Recording stopped")

        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if records:
    file_exists = os.path.exists(OUTPUT)

    # Append new recordings so each action can be collected
    with open(OUTPUT, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(OUTPUT) == 0:
            writer.writerow(columns)

        writer.writerows(records)

    print("Saved:", OUTPUT)
    print("New frames:", len(records))
else:
    print("No labeled frames recorded")
