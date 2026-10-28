
import csv
import os
from datetime import datetime

# ==========================================
# 1. FILE PATH
# ==========================================

LOG_FILE = "outputs/robot_action_log.csv"

os.makedirs("outputs", exist_ok=True)


# ==========================================
# 2. LOG DETECTED ROBOT ACTION
# ==========================================

def log_action(action):
    file_exists = (
        os.path.exists(LOG_FILE)
        and os.path.getsize(LOG_FILE) > 0
    )

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        # Write column headings for a new file
        if not file_exists:
            writer.writerow([
                "timestamp",
                "detected_action"
            ])

        # Save timestamp and action
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            action
        ])

    print("Action logged:", action)


# ==========================================
# 3. TEST THE LOGGER
# ==========================================

if __name__ == "__main__":
    print("RoboMind Action Logger Test")

    log_action("raise_hand")
    log_action("lower_hand")
    log_action("bend_arm")

    print("\nLog saved to:", LOG_FILE)

