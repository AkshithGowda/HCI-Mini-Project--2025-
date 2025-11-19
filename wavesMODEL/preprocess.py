import pandas as pd
import numpy as np
import scipy.signal as signal

# Load dataset (update path if needed)
df = pd.read_csv("../datasets/emotions.csv")

# Convert all to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Remove rows that are mostly NaN
df = df.dropna(thresh=int(df.shape[1] * 0.9))

# Signal sample rate (approx for EEG)
fs = 256

# EEG bands
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 45)
}

def compute_bandpower(row):
    freqs, psd = signal.welch(row, fs)
    band_feats = {}
    for name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_feats[name] = np.trapz(psd[idx], freqs[idx])
    return band_feats

# Compute bandpowers for each row
features = df.apply(lambda r: compute_bandpower(r.values), axis=1, result_type='expand')

# Add 4 statistical extras
features["f1"] = df.mean(axis=1)
features["f2"] = df.std(axis=1)
features["f3"] = df.max(axis=1)
features["f4"] = df.min(axis=1)

# Save final 9-feature dataset
features.to_csv("features_epoched.csv", index=False)

print("Processed", len(features), "rows → features_epoched.csv created!")
