
import streamlit as st
import pandas as pd
import pickle
import os
import subprocess
import sys
from collections import Counter

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="RoboMind V4 Dashboard",
    page_icon="🤖",
    layout="wide"
)

# =====================================
# V4 FILE PATHS
# =====================================

DATA_FILE = "data/processed/labeled_data_v4.csv"
MODEL_FILE = "models/action_model_v4.pkl"
SIMULATION_FILE = "robot_simulation_v4.py"
LOG_FILE = "outputs/robot_action_log_v4.csv"

# =====================================
# HEADER
# =====================================

st.title("🤖 RoboMind V4")
st.subheader("AI-Based Human Action Recognition and Robot Simulation")

st.write(
    "A computer vision and machine learning system that "
    "recognizes human arm actions and maps them to a virtual robot."
)

# =====================================
# LOAD DATASET
# =====================================

rows = []

if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        rows = df.to_dict("records")
    except Exception as error:
        st.error(f"Could not read the dataset: {error}")
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

# =====================================
# LOAD MODEL STATUS
# =====================================

model_exists = os.path.exists(MODEL_FILE)
simulation_exists = os.path.exists(SIMULATION_FILE)
log_exists = os.path.exists(LOG_FILE)

# =====================================
# SIDEBAR NAVIGATION
# =====================================

st.sidebar.title("RoboMind Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Dataset Analysis",
        "Model Information",
        "Evaluation Results",
        "Action History",
        "Run Robot Simulation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("**Project:** RoboMind V4")
st.sidebar.write("**Algorithm:** K-Nearest Neighbors")
st.sidebar.write("**Pose Detection:** MediaPipe")
st.sidebar.write("**Interface:** Streamlit")

# =====================================
# OVERVIEW
# =====================================

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Training Frames", len(df))

    with col2:
        if "action" in df.columns:
            action_classes = df["action"].nunique()
        else:
            action_classes = 0

        st.metric("Action Classes", action_classes)

    with col3:
        st.metric(
            "V4 Model",
            "Available" if model_exists else "Not Found"
        )

    with col4:
        st.metric(
            "Simulation",
            "Available" if simulation_exists else "Not Found"
        )

    st.markdown("---")
    st.subheader("Recognized Actions")

    a, b, c = st.columns(3)

    with a:
        st.info("🙋 Raise Hand")
        st.write("Recognizes an arm raised upward.")

    with b:
        st.info("⬇️ Lower Hand")
        st.write("Recognizes an arm lowered downward.")

    with c:
        st.info("💪 Bend Arm")
        st.write("Recognizes a bent elbow position.")

    st.markdown("---")
    st.subheader("System Workflow")

    st.code(
        "Webcam Input\n"
        "    ↓\n"
        "MediaPipe Pose Detection\n"
        "    ↓\n"
        "Four Arm-Angle Features\n"
        "    ↓\n"
        "Feature Normalization\n"
        "    ↓\n"
        "KNN Action Classification\n"
        "    ↓\n"
        "2D Robot Arm Simulation\n"
        "    ↓\n"
        "CSV Action Logging",
        language="text"
    )

    st.success(
        "RoboMind V4 uses pose estimation and KNN "
        "to map human arm movements to simulated robot actions."
    )

# =====================================
# DATASET ANALYSIS
# =====================================

elif page == "Dataset Analysis":

    st.header("📊 V4 Dataset Analysis")

    if not df.empty and "action" in df.columns:

        st.metric("Total Training Frames", len(df))

        counts = df["action"].value_counts()

        st.subheader("Action Distribution")
        st.bar_chart(counts)

        st.subheader("Action Counts")

        col1, col2, col3 = st.columns(3)

        action_info = [
            ("raise_hand", col1),
            ("lower_hand", col2),
            ("bend_arm", col3)
        ]

        for action, col in action_info:
            with col:
                st.metric(
                    action.replace("_", " ").title(),
                    int(counts.get(action, 0))
                )

        st.markdown("---")
        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download V4 Training Dataset",
            data=df.to_csv(index=False),
            file_name="labeled_data_v4.csv",
            mime="text/csv"
        )

        st.caption(
            "Each row represents a labeled training frame. "
            "Frames from the same recording may be correlated."
        )

    else:
        st.warning(
            "V4 dataset is empty or does not contain an 'action' column. "
            "Check data/processed/labeled_data_v4.csv."
        )

# =====================================
# MODEL INFORMATION
# =====================================

elif page == "Model Information":

    st.header("🧠 V4 Model Information")

    if model_exists:

        try:
            with open(MODEL_FILE, "rb") as file:
                saved = pickle.load(file)

            # V4 model is stored as a dictionary
            model = saved["model"]
            features = saved["features"]

            st.success("V4 trained model loaded successfully.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Algorithm", "KNN")

            with col2:
                st.metric(
                    "Neighbors (K)",
                    getattr(model, "n_neighbors", "Unknown")
                )

            with col3:
                st.metric("Input Features", len(features))

            st.markdown("---")
            st.subheader("Model File")

            st.write(f"**Path:** `{MODEL_FILE}`")
            st.write(f"**Classifier:** `{type(model).__name__}`")

            st.subheader("Input Features")

            for feature in features:
                st.write(f"- `{feature}`")

            st.subheader("Stored Normalization Limits")

            mins = saved.get("mins")
            maxs = saved.get("maxs")

            if mins is not None and maxs is not None:
                limits = pd.DataFrame({
                    "Feature": features,
                    "Minimum": list(mins),
                    "Maximum": list(maxs)
                })

                st.dataframe(
                    limits,
                    use_container_width=True,
                    hide_index=True
                )

            st.info(
                "The V4 model uses four arm-angle features and "
                "KNN classification. This page reads the existing "
                "model without retraining or modifying it."
            )

        except Exception as error:
            st.error(f"Could not load the V4 model: {error}")

    else:
        st.error(f"Model not found: {MODEL_FILE}")

# =====================================
# EVALUATION RESULTS
# =====================================

elif page == "Evaluation Results":

    st.header("📈 V4 Model Evaluation")

    st.info(
        "Test and validation results come from different evaluation "
        "setups. They must not be treated as directly comparable."
    )

    # Test-set experiment
    st.subheader("Test Dataset — 694 Samples")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Original KNN Accuracy", "72.05%")

    with col2:
        st.metric("Experimental Rule Accuracy", "81.41%")

    with col3:
        st.metric("Best Tested Threshold", "30°")

    st.write(
        "The 81.41% result came from applying a shoulder-angle rule "
        "to the test predictions. This rule was not retained because "
        "it reduced accuracy on the separate validation experiment."
    )

    st.markdown("---")

    # Validation-set experiment
    st.subheader("Validation Dataset — 150 Samples")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Original KNN Accuracy", "92.67%")

    with col2:
        st.metric("Rule-Adjusted Accuracy", "88.00%")

    with col3:
        st.metric("Correct Original Predictions", "139 / 150")

    st.write(
        "The original KNN performed better than the shoulder-angle "
        "rule on this validation split. Therefore, keep the original "
        "V4 model and do not apply the experimental rule."
    )

    st.markdown("---")
    st.subheader("Original KNN Validation Confusion Matrix")

    confusion = pd.DataFrame(
        [
            [45, 2, 0],
            [2, 51, 0],
            [5, 2, 43]
        ],
        index=["Actual bend_arm", "Actual lower_hand", "Actual raise_hand"],
        columns=[
            "Predicted bend_arm",
            "Predicted lower_hand",
            "Predicted raise_hand"
        ]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )

    st.caption(
        "Rows represent actual actions and columns represent predicted "
        "actions. The matrix is from the 150-sample validation experiment."
    )

    st.subheader("Original KNN Validation Classification Report")

    report = pd.DataFrame(
        {
            "Action": ["bend_arm", "lower_hand", "raise_hand"],
            "Precision": [0.87, 0.93, 1.00],
            "Recall": [0.96, 0.96, 0.86],
            "F1-score": [0.91, 0.94, 0.92],
            "Support": [47, 53, 50]
        }
    )

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

# =====================================
# ACTION HISTORY
# =====================================

elif page == "Action History":

    st.header("📋 V4 Action History")

    if os.path.exists(LOG_FILE):

        try:
            log_df = pd.read_csv(LOG_FILE)

            if log_df.empty:
                st.info("The V4 action log exists but has no records.")

            elif not {"timestamp", "action"}.issubset(log_df.columns):
                st.warning(
                    "The log must contain 'timestamp' and 'action' columns."
                )

            else:
                st.subheader("Session Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Logged Actions", len(log_df))

                with col2:
                    st.metric(
                        "Unique Actions",
                        log_df["action"].nunique()
                    )

                with col3:
                    st.metric(
                        "Latest Action",
                        str(log_df.iloc[-1]["action"])
                    )

                st.markdown("---")
                st.subheader("Action Counts")

                action_counts = log_df["action"].value_counts()
                st.bar_chart(action_counts)

                st.subheader("Recent Action Records")

                st.dataframe(
                    log_df.tail(50).iloc[::-1],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download V4 Action Log",
                    data=log_df.to_csv(index=False),
                    file_name="robot_action_log_v4.csv",
                    mime="text/csv"
                )

        except Exception as error:
            st.error(f"Could not read the action log: {error}")

    else:
        st.warning(f"Action log not found: {LOG_FILE}")
        st.write(
            "Run the V4 simulation and perform some actions to create log records."
        )

# =====================================
# RUN V4 ROBOT SIMULATION
# =====================================

elif page == "Run Robot Simulation":

    st.header("🤖 Launch RoboMind V4 Simulation")

    st.write(
        "Launch live webcam-based human action recognition "
        "and the 2D robot simulation in a separate desktop window."
    )

    st.warning(
        "This feature requires a local computer with webcam access. "
        "The simulation opens outside the Streamlit browser page."
    )

    if st.button("▶ Start V4 Robot Simulation"):

        if not simulation_exists:
            st.error(
                f"Simulation file not found: {SIMULATION_FILE}"
            )

        else:
            try:
                subprocess.Popen(
                    [sys.executable, os.path.abspath(SIMULATION_FILE)],
                    cwd=os.getcwd()
                )

                st.success(
                    "Launch requested. Check your desktop for the "
                    "V4 webcam and robot simulation window."
                )

            except Exception as error:
                st.error(f"Could not launch simulation: {error}")

    st.markdown("---")
    st.subheader("How to use")

    st.write("1. Click **Start V4 Robot Simulation**.")
    st.write("2. Perform `raise_hand`, `lower_hand`, and `bend_arm`.")
    st.write("3. Check the predicted action and robot movement.")
    st.write("4. Return to Action History to inspect the CSV log.")

    st.caption(
        "To stop the simulation, follow the quit instructions "
        "implemented in robot_simulation_v4.py."
    )
