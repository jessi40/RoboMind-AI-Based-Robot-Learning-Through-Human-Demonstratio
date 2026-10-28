
import pandas as pd
import pickle

# Load V5 model
with open("models/action_model_v5.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
features = saved["features"]
mins = saved["mins"]
maxs = saved["maxs"]

# Load unseen test data
df = pd.read_csv("data/processed/test_data_v4.csv")

X = df[features].values
y_true = df["action"].values

# Apply training normalization
ranges = maxs - mins
ranges[ranges == 0] = 1
X_scaled = (X - mins) / ranges

# Predict
y_pred = model.predict(X_scaled)

# Overall accuracy
correct = sum(y_true == y_pred)
print("V5 TEST RESULTS")
print("----------------")
print(f"Correct: {correct}/{len(y_true)}")
print(f"Accuracy: {correct / len(y_true) * 100:.2f}%")

# Accuracy for each action
for action in sorted(set(y_true)):
    mask = y_true == action
    total = sum(mask)
    action_correct = sum(y_pred[mask] == y_true[mask])
    print(
        f"{action}: {action_correct}/{total} "
        f"({action_correct / total * 100:.2f}%)"
    )

# Confusion matrix
actions = sorted(set(y_true) | set(y_pred))
print("\nConfusion Matrix")
print("Rows = Actual; Columns = Predicted")
print("Actions:", actions)

for action in actions:
    row = [
        sum((y_true == action) & (y_pred == predicted))
        for predicted in actions
    ]
    print(action, row)

