# plot_final_results_fixed.py
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib

# Use TkAgg backend (best for Windows live plotting)
matplotlib.use("TkAgg")

print("Loading mental_state_bandpower.csv...")
df = pd.read_csv("mental_state_bandpower.csv")

print("Loading model.pkl...")
model = joblib.load("model.pkl")

X = df.drop(columns=["label"])
y_true = df["label"]
y_pred = model.predict(X)

# ---------------------------
# 1. Print Evaluation
# ---------------------------
print("\n=== Evaluation ===")
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print(classification_report(y_true, y_pred, digits=3))
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# ============================================================
# 🔵 GRAPH 1: CONFUSION MATRIX
# ============================================================
plt.figure(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["OFF (0)", "ON (1)"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Brainwave Classification")
plt.tight_layout()
plt.show(block=True)   # 🔥 ensures window doesn't close

# ============================================================
# 🔴 GRAPH 2: ACTIVATION SCORE DISTRIBUTION
# ============================================================
plt.figure(figsize=(8,4))
df["score"].plot(kind="hist", bins=20, color="violet", edgecolor="black")
plt.title("Activation Score Distribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.tight_layout()
plt.show(block=True)

# ============================================================
# 🟢 GRAPH 3: LABEL DISTRIBUTION (ON/OFF)
# ============================================================
plt.figure(figsize=(6,4))
df["label"].value_counts().plot(
    kind="bar",
    color=["lightgreen", "salmon"],
    rot=0
)
plt.title("Label Distribution (ON / OFF)")
plt.xlabel("Label")
plt.ylabel("Samples")
plt.tight_layout()
plt.show(block=True)

# ============================================================
# 🟣 GRAPH 4: FEATURE IMPORTANCE
# ============================================================
if hasattr(model, "feature_importances_"):
    import numpy as np

    importance = model.feature_importances_
    names = X.columns

    idx = np.argsort(importance)[-10:]  # top 10 important features

    plt.figure(figsize=(8,5))
    plt.barh(names[idx], importance[idx], color="skyblue")
    plt.title("Top 10 Most Important EEG Features")
    plt.tight_layout()
    plt.show(block=True)
else:
    print("\n⚠️ Model does not support feature importances.")
