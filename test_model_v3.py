
import cv2
import mediapipe as mp
import pickle
import math

MODEL_PATH = "models/action_model_v3.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

train_X = model["train_X"]
train_y = model["train_y"]
features = model["features"]
mins = model["mins"]
maxs = model["maxs"]
k = model["k"]


def angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    lengths = math.hypot(*ba) * math.hypot(*bc)

    if lengths == 0:
        return None

    cosine = max(-1, min(1, dot / lengths))
    return math.degrees(math.acos(cosine))


def extract_features(landmarks):
    p = landmarks.landmark

    return [
        angle((p[11].x, p[11].y),
              (p[13].x, p[13].y),
              (p[15].x, p[15].y)),

        angle((p[12].x, p[12].y),
              (p[14].x, p[14].y),
              (p[16].x, p[16].y)),

        angle((p[23].x, p[23].y),
              (p[11].x, p[11].y),
              (p[13].x, p[13].y)),

        angle((p[24].x, p[24].y),
              (p[12].x, p[12].y),
              (p[14].x, p[14].y))
    ]


def normalize(values):
    result = []

    for i, value in enumerate(values):
        span = maxs[i] - mins[i]
        result.append(
            (value - mins[i]) / span if span else 0.0
        )

    return result


def predict(sample):
    distances = []

    for x, label in zip(train_X, train_y):
        distance = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(sample, x))
        )
        distances.append((distance, label))

    distances.sort(key=lambda item: item[0])
    nearest = [label for _, label in distances[:k]]

    return max(set(nearest), key=nearest.count)


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    print("Press Q to quit.")

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Cannot access webcam.")
            break

        result = pose.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        prediction = "No pose detected"

        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            values = extract_features(result.pose_landmarks)

            if all(v is not None for v in values):
                prediction = predict(normalize(values))

        cv2.putText(
            frame,
            "Prediction: " + prediction,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("RoboMind - Model V3 Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
