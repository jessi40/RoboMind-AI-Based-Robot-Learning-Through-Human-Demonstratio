
import cv2
import mediapipe as mp
import math
import pickle
from collections import Counter

# Load the trained model
with open("models/action_model.pkl", "rb") as file:
    saved = pickle.load(file)

train_X = saved["train_X"]
train_y = saved["train_y"]
mins = saved["mins"]
maxs = saved["maxs"]
k = saved["k"]

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Calculate angle between three points
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    lengths = math.hypot(*ba) * math.hypot(*bc)

    if lengths == 0:
        return None

    cosine = max(-1, min(1, dot / lengths))
    return math.degrees(math.acos(cosine))


# Extract the same four features used during training
def extract_features(landmarks):
    def point(index):
        p = landmarks[index]
        return (p.x, p.y)

    left_shoulder = point(11)
    left_elbow = point(13)
    left_wrist = point(15)
    right_shoulder = point(12)
    right_elbow = point(14)
    right_wrist = point(16)
    left_hip = point(23)
    right_hip = point(24)

    left_elbow_angle = calculate_angle(
        left_shoulder, left_elbow, left_wrist
    )
    right_elbow_angle = calculate_angle(
        right_shoulder, right_elbow, right_wrist
    )
    left_shoulder_angle = calculate_angle(
        left_elbow, left_shoulder, left_hip
    )
    right_shoulder_angle = calculate_angle(
        right_elbow, right_shoulder, right_hip
    )

    return [
        left_elbow_angle,
        right_elbow_angle,
        left_shoulder_angle,
        right_shoulder_angle
    ]


# Normalize features using training ranges
def normalize(values):
    result = []

    for i, value in enumerate(values):
        span = maxs[i] - mins[i]
        result.append(
            (value - mins[i]) / span if span else 0.0
        )

    return result


# Predict using K-nearest neighbors
def predict_action(values):
    sample = normalize(values)
    distances = []

    for train_sample, label in zip(train_X, train_y):
        distance = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(sample, train_sample)
            )
        )
        distances.append((distance, label))

    nearest = sorted(distances)[:min(k, len(distances))]
    votes = Counter(label for _, label in nearest)

    return votes.most_common(1)[0][0]


# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit(
        "Cannot open webcam. Close other camera apps and try again."
    )

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        success, frame = cap.read()

        if not success:
            print("Cannot read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        label = "No person detected"

        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            values = extract_features(
                result.pose_landmarks.landmark
            )

            if all(v is not None for v in values):
                label = predict_action(values)

        cv2.putText(
            frame,
            "Action: " + label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("RoboMind - Live Action Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
