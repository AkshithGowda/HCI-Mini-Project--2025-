import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.utils import resample
from pathlib import Path

# =========================
# SAME FEATURE ENGINEERING AS TRAINING
# raw must be: delta, theta, alpha, beta, gamma, f1, f2, f3, f4
# =========================
def compute_features_array(raw):
    delta, theta, alpha, beta, gamma, f1, f2, f3, f4 = raw

    beta_alpha = beta / (alpha + 1e-6)
    gamma_power = gamma
    theta_alpha = theta / (alpha + 1e-6)
    stress_index = (beta + gamma) / (alpha + 1e-6)
    engagement_index = beta / (alpha + theta + 1e-6)

    # Single-frame heuristics (same as GUI/training logic)
    is_focus = int(beta_alpha > 1.2)
    is_memory = int(gamma_power > gamma_power)  # placeholder consistency
    is_relaxed = int(theta_alpha < 1.0)
    is_stressed = int(stress_index > 1.3)
    is_engaged = int(engagement_index > 1.0)

    score = is_focus + is_memory + is_relaxed + is_stressed + is_engaged

    return {
        "delta": delta, "theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma,
        "f1": f1, "f2": f2, "f3": f3, "f4": f4,
        "beta_alpha": beta_alpha, "gamma_power": gamma_power, "theta_alpha": theta_alpha,
        "stress_index": stress_index, "engagement_index": engagement_index,
        "is_focus": is_focus, "is_memory": is_memory, "is_relaxed": is_relaxed,
        "is_stressed": is_stressed, "is_engaged": is_engaged, "score": score
    }

# =========================
# HELPER: build processed feature DF from any wide CSV
# It will try to map columns by name; if not found, it will take first 9 numeric columns
# =========================
def make_processed_features(df):
    wanted = ["delta","theta","alpha","beta","gamma","f1","f2","f3","f4"]
    lower_cols = [c.lower() for c in df.columns]

    name_map = {}
    for w in wanted:
        if w in lower_cols:
            name_map[w] = df.columns[lower_cols.index(w)]

    if len(name_map) == 9:
        base = df[[name_map[w] for w in wanted]].to_numpy()
    else:
        # fallback: take first 9 numeric columns
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 9:
            raise ValueError("Could not find 9 numeric columns for features.")
        base = num_df.iloc[:, :9].to_numpy()
        print("⚠️ Using first 9 numeric columns as features (column names didn’t match delta..f4).")

    processed = [compute_features_array(row) for row in base]
    return pd.DataFrame(processed)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    np.random.seed(42)

    data_path = Path("mental-state.csv")
    if not data_path.exists():
        raise SystemExit("mental-state.csv not found in current folder.")

    # Load new dataset
    raw_df = pd.read_csv(data_path)
    print(f"\nLoaded NEW dataset: shape={raw_df.shape}")
    print(raw_df.head(3))

    # Build engineered feature table (exactly like training)
    X_all = make_processed_features(raw_df)

    # Load trained model
    model = joblib.load("model.pkl")

    # Map labels to ON/OFF if present; else derive from score
    label_col = None
    for cand in ["Label","label","target","Target","y","Y"]:
        if cand in raw_df.columns:
            label_col = cand
            break

    if label_col:
        y_raw = raw_df[label_col]
        # If numeric with values like 1.0/2.0, map: ON=1, OFF=else (simple, consistent)
        if pd.api.types.is_numeric_dtype(y_raw):
            y_true = (y_raw == y_raw.min()).astype(int)  # assume lower code = "more active" class
            # If you know semantics (e.g., 1=ON, 2=OFF), you can hardcode:
            # y_true = (y_raw == 1).astype(int)
        else:
            # Try simple mapping of strings
            mapping = {"on":1,"off":0,"yes":1,"no":0,"high":1,"low":0,"active":1,"calm":0}
            y_true = y_raw.astype(str).str.lower().map(mapping).fillna(0).astype(int)
    else:
        print("\nℹ️ 'Label' column not found. Using engineered activation score: label = (score >= 3).")
        y_true = (X_all["score"] >= 3).astype(int)

    # ---- Evaluate on full (imbalanced) set
    try:
        prob_all = model.predict_proba(X_all)[:,1]
    except Exception:
        prob_all = model.predict(X_all).astype(float)
    y_pred_all = (prob_all >= 0.5).astype(int)

    print("\n===== FULL SET (likely imbalanced) =====")
    print("Accuracy:", round(accuracy_score(y_true, y_pred_all), 4))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_all, average="binary", zero_division=0)
    print(f"Precision (ON): {p:.3f} | Recall (ON): {r:.3f} | F1 (ON): {f1:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_all))
    try:
        print("ROC-AUC:", round(roc_auc_score(y_true, prob_all), 4))
    except Exception:
        pass

    # ---- Build a balanced evaluation subset for fair performance
    df_eval = X_all.copy()
    df_eval["y_true"] = y_true.values
    cls0 = df_eval[df_eval["y_true"] == 0]
    cls1 = df_eval[df_eval["y_true"] == 1]

    if len(cls0) == 0 or len(cls1) == 0:
        print("\n⚠️ Cannot build a balanced set: one of the classes is missing.")
    else:
        n = min(len(cls0), len(cls1), 2000)  # cap to keep it fast
        cls0_b = resample(cls0, n_samples=n, random_state=42, replace=False)
        cls1_b = resample(cls1, n_samples=n, random_state=42, replace=(len(cls1) < n))
        balanced = pd.concat([cls0_b, cls1_b], ignore_index=True).sample(frac=1, random_state=42)

        yb_true = balanced["y_true"].astype(int).values
        Xb = balanced.drop(columns=["y_true"])

        try:
            prob_b = model.predict_proba(Xb)[:,1]
        except Exception:
            prob_b = model.predict(Xb).astype(float)
        yb_pred_05 = (prob_b >= 0.5).astype(int)

        print("\n===== BALANCED SUBSET (fair check) =====")
        print("Accuracy:", round(accuracy_score(yb_true, yb_pred_05), 4))
        p, r, f1, _ = precision_recall_fscore_support(yb_true, yb_pred_05, average="binary", zero_division=0)
        print(f"Precision (ON): {p:.3f} | Recall (ON): {r:.3f} | F1 (ON): {f1:.3f}")
        print("Confusion Matrix:\n", confusion_matrix(yb_true, yb_pred_05))
        try:
            print("ROC-AUC:", round(roc_auc_score(yb_true, prob_b), 4))
        except Exception:
            pass

        # ---- Optional: tune threshold for best F1 on balanced set
        thresholds = np.linspace(0.1, 0.9, 17)
        best = (0.0, 0.5)  # (best_f1, thresh)
        for t in thresholds:
            pred_t = (prob_b >= t).astype(int)
            _, _, f1_t, _ = precision_recall_fscore_support(yb_true, pred_t, average="binary", zero_division=0)
            if f1_t > best[0]:
                best = (f1_t, float(t))
        best_f1, best_t = best
        pred_best = (prob_b >= best_t).astype(int)
        p_b, r_b, f1_b, _ = precision_recall_fscore_support(yb_true, pred_best, average="binary", zero_division=0)

        print(f"\n>>> Threshold tuning on balanced set: best F1={best_f1:.3f} at threshold={best_t:.2f}")
        print(f"Precision: {p_b:.3f} | Recall: {r_b:.3f} | F1: {f1_b:.3f}")
