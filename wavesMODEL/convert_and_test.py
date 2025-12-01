# convert_and_test.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("🔄 Loading mental-state.csv ...")
df = pd.read_csv("mental-state.csv")
print("Loaded shape =", df.shape)

# --------------------------------------------------------------
# 1) IDENTIFY FREQUENCY BIN COLUMNS
# --------------------------------------------------------------
# We search for columns like: freq_100_0, freq_200_1, etc.
freq_cols = [c for c in df.columns if "freq" in c.lower()]

if len(freq_cols) == 0:
    raise ValueError("❌ No frequency-bin columns found!")

print(f"Found {len(freq_cols)} frequency bin columns")

# Extract frequencies from column names: freq_750_3 → 750 Hz
def extract_freq(col):
    try:
        parts = col.replace("freq_", "").split("_")
        return float(parts[0])
    except:
        return None

freq_map = {col: extract_freq(col) for col in freq_cols}

# --------------------------------------------------------------
# 2) DEFINE EEG BAND RANGES
# --------------------------------------------------------------
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45)
}

# --------------------------------------------------------------
# 3) COMPUTE BANDPOWERS BY AVERAGING FREQUENCY BINS
# --------------------------------------------------------------
band_values = {b: [] for b in bands}

freq_columns_used = []

for col, f in freq_map.items():
    if f is None:
        continue
    for band_name, (low, high) in bands.items():
        if low <= f <= high:
            band_values[band_name].append(col)
            freq_columns_used.append(col)
            break

print("\n📡 Columns mapped per band:")
for b in bands:
    print(f"{b}: {len(band_values[b])} columns")

# --------------------------------------------------------------
# 4) COMPUTE BANDPOWER FEATURES
# --------------------------------------------------------------
out = pd.DataFrame()
for band in bands:
    if len(band_values[band]) == 0:
        out[band] = np.zeros(len(df))
    else:
        out[band] = df[band_values[band]].mean(axis=1)

# Add simple stats like f1, f2, f3, f4
all_freq_matrix = df[freq_columns_used].values
out["f1"] = np.mean(all_freq_matrix, axis=1)
out["f2"] = np.std(all_freq_matrix, axis=1)
out["f3"] = np.max(all_freq_matrix, axis=1)
out["f4"] = np.min(all_freq_matrix, axis=1)

# --------------------------------------------------------------
# 5) ENGINEERED FEATURES (same as your training pipeline)
# --------------------------------------------------------------
out["beta_alpha"] = out["beta"] / (out["alpha"] + 1e-6)
out["gamma_power"] = out["gamma"]
out["theta_alpha"] = out["theta"] / (out["alpha"] + 1e-6)
out["stress_index"] = (out["beta"] + out["gamma"]) / (out["alpha"] + 1e-6)
out["engagement_index"] = out["beta"] / (out["alpha"] + out["theta"] + 1e-6)

out["is_focus"] = (out["beta_alpha"] > 1.2).astype(int)
out["is_memory"] = (out["gamma_power"] > out["gamma_power"]).astype(int)  # your original definition
out["is_relaxed"] = (out["theta_alpha"] < 1.0).astype(int)
out["is_stressed"] = (out["stress_index"] > 1.3).astype(int)
out["is_engaged"] = (out["engagement_index"] > 1.0).astype(int)

# Activation score
out["score"] = out[[
    "is_focus", "is_memory", "is_relaxed",
    "is_stressed", "is_engaged"
]].sum(axis=1)

# Convert score → ON/OFF labels
out["label"] = (out["score"] >= 3).astype(int)

print("\n✅ Finished feature conversion.")
print("New feature set shape:", out.shape)

# --------------------------------------------------------------
# 6) LOAD MODEL
# --------------------------------------------------------------
model = joblib.load("model.pkl")
print("🔧 Loaded model.pkl")

# --------------------------------------------------------------
# 7) PREDICT AND EVALUATE
# --------------------------------------------------------------
X = out.drop(columns=["label"])
y_true = out["label"]

y_pred = model.predict(X)

print("\n🎯 REAL ACCURACY ON mental-state.csv AFTER BAND CONVERSION:")
print("Accuracy =", round(accuracy_score(y_true, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save processed dataset
out.to_csv("mental_state_bandpower.csv", index=False)
print("\n💾 Saved processed dataset → mental_state_bandpower.csv")
