
import cv2
import mediapipe as mp
import csv
import os
import time

# Create folder for demonstration data
os.makedirs("data/raw", exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

records = []
recording = False
frame_id = 0
start_time = 0

print("Press R to start or stop recording")
print("Press Q to quit")

try:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Cannot access webcam")
            break

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect body landmarks
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            if recording:
                row = {
                    "frame": frame_id,
                    "time": round(time.time() - start_time, 3)
                }

                for i, point in enumerate(
                    results.pose_landmarks.landmark
                ):
                    row[f"x{i}"] = point.x
                    row[f"y{i}"] = point.y
                    row[f"z{i}"] = point.z
                    row[f"visibility{i}"] = point.visibility

                records.append(row)
                frame_id += 1

        status = "RECORDING" if recording else "READY"

        cv2.putText(
            frame, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.imshow("RoboMind - Demonstration Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            recording = not recording

            if recording:
                start_time = time.time()
                print("Recording started")
            else:
                print("Recording stopped")

        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# Save data using Python's built-in CSV module
if records:
    filename = "data/raw/demonstration.csv"

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)

    print("Data saved successfully!")
    print("File:", filename)
    print("Total frames:", len(records))
else:
    print("No data recorded")
