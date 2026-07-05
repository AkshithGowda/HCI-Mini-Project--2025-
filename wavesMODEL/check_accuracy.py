import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load dataset
df = pd.read_csv("features_epoched.csv")

# -----------------------------------------------
# SAME SCIENTIFIC EEG FEATURE ENGINEERING AS TRAINING
# -----------------------------------------------

df["beta_alpha"] = df["beta"] / (df["alpha"] + 1e-6)
df["gamma_power"] = df["gamma"]
df["theta_alpha"] = df["theta"] / (df["alpha"] + 1e-6)
df["stress_index"] = (df["beta"] + df["gamma"]) / (df["alpha"] + 1e-6)
df["engagement_index"] = df["beta"] / (df["alpha"] + df["theta"] + 1e-6)

df["is_focus"] = (df["beta_alpha"] > df["beta_alpha"].mean()).astype(int)
df["is_memory"] = (df["gamma_power"] > df["gamma_power"].mean()).astype(int)
df["is_relaxed"] = (df["theta_alpha"] < df["theta_alpha"].mean()).astype(int)
df["is_stressed"] = (df["stress_index"] > df["stress_index"].mean()).astype(int)
df["is_engaged"] = (df["engagement_index"] > df["engagement_index"].mean()).astype(int)

df["score"] = (
    df["is_focus"] +
    df["is_memory"] +
    df["is_relaxed"] +
    df["is_stressed"] +
    df["is_engaged"]
)

df["label"] = (df["score"] >= 3).astype(int)

# -----------------------------------------------
# Evaluate model
# -----------------------------------------------

X = df.drop("label", axis=1)
y_true = df["label"]

model = joblib.load("model.pkl")
y_pred = model.predict(X)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
