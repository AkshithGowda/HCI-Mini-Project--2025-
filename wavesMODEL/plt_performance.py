import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("features_epoched.csv")
df["label"] = (df["beta"] > df["beta"].median()).astype(int)

model = joblib.load("model.pkl")

X = df.drop("label", axis=1)
y = df["label"]
pred = model.predict(X)

plt.figure(figsize=(12,4))
plt.plot(y[:200], label="Actual")
plt.plot(pred[:200], label="Predicted")
plt.legend()
plt.title("Prediction Timeline (First 200 samples)")
plt.xlabel("Sample Index")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("prediction_timeline.png")

print("Saved prediction_timeline.png")
