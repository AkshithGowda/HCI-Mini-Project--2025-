import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load features
df = pd.read_csv("features_epoched.csv")

# -----------------------------------------------
# SCIENTIFIC EEG LABELING (Realistic)
# -----------------------------------------------

# 1. EEG scientific indexes
df["beta_alpha"] = df["beta"] / (df["alpha"] + 1e-6)
df["gamma_power"] = df["gamma"]
df["theta_alpha"] = df["theta"] / (df["alpha"] + 1e-6)       # relaxation indicator
df["stress_index"] = (df["beta"] + df["gamma"]) / (df["alpha"] + 1e-6)
df["engagement_index"] = df["beta"] / (df["alpha"] + df["theta"] + 1e-6)

# 2. Convert indexes into ON/OFF indicators
df["is_focus"] = (df["beta_alpha"] > df["beta_alpha"].mean()).astype(int)
df["is_memory"] = (df["gamma_power"] > df["gamma_power"].mean()).astype(int)
df["is_relaxed"] = (df["theta_alpha"] < df["theta_alpha"].mean()).astype(int)  # lower = more active
df["is_stressed"] = (df["stress_index"] > df["stress_index"].mean()).astype(int)
df["is_engaged"] = (df["engagement_index"] > df["engagement_index"].mean()).astype(int)

# 3. Final ON/OFF score
df["score"] = (
    df["is_focus"] +
    df["is_memory"] +
    df["is_relaxed"] +
    df["is_stressed"] +
    df["is_engaged"]
)

# ON if at least 3 EEG conditions indicate cognitive activation
df["label"] = (df["score"] >= 3).astype(int)

# ------------------------------------------------
# ML Training
# ------------------------------------------------

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
