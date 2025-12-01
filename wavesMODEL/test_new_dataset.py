import pandas as pd
import joblib
import sys, os
sys.path.append(os.path.dirname(__file__))
from upgraded_gui import compute_features_array
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load your new dataset
# -----------------------------
df = pd.read_csv("mental-state.csv")

print("\nLoaded NEW dataset:")
print(df.head())

# -----------------------------
# Convert each row to your model's feature format
# -----------------------------
processed_rows = []

for idx, row in df.iterrows():
    # Extract the 9 raw EEG features
    raw = row.values[:9]   # FIRST 9 columns must be delta..f4
    processed = compute_features_array(raw).iloc[0]
    processed_rows.append(processed)

processed_df = pd.DataFrame(processed_rows)

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# If the mental-state dataset has its own labels:
# rename the label column to "label"
# -----------------------------
if "label" in df.columns:
    y_true = df["label"]
else:
    print("\n❗ No labels found in mental-state.csv")
    print("➡️ I will generate my own ON/OFF labels based on activation score")
    y_true = (processed_df["score"] >= 3).astype(int)

# -----------------------------
# Predict using trained model
# -----------------------------
y_pred = model.predict(processed_df)

# -----------------------------
# Results
# -----------------------------
print("\n⭐ Accuracy on NEW DATASET:", accuracy_score(y_true, y_pred))
print("\n⭐ Classification Report:\n", classification_report(y_true, y_pred))
