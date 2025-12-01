# wavesMODEL/test_eye_state.py
import pandas as pd, joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("eye_state_features.csv")
model = joblib.load("model.pkl")

X = df.drop(columns=["eye_label","score_label"])
y_score = df["score_label"]  # comparable to your training labels
y_eye = df["eye_label"]      # optional: open/closed from UCI

# Evaluate vs score_label
pred = model.predict(X)
print("\n=== Evaluation vs score_label (comparable to your training) ===")
print("Accuracy:", round(accuracy_score(y_score, pred), 3))
print(classification_report(y_score, pred, digits=3))
print("Confusion:\n", confusion_matrix(y_score, pred))

# Optional: also see how it relates to eye open/closed
pred_eye = model.predict(X)
print("\n=== (Optional) Evaluation vs eye_label (0=open, 1=closed) ===")
print("Accuracy:", round(accuracy_score(y_eye, pred_eye), 3))
print(classification_report(y_eye, pred_eye, digits=3))
print("Confusion:\n", confusion_matrix(y_eye, pred_eye))
