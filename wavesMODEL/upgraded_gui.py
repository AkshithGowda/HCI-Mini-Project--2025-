# upgraded_gui_live.py  (with Gradient Focus Bar + Prediction Smoothing)
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import joblib
import threading
import time
from collections import deque
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# ---------------------------
# Feature computation (same as train.py)
# ---------------------------
def compute_features_array(raw):
    delta, theta, alpha, beta, gamma, f1, f2, f3, f4 = raw

    beta_alpha = beta / (alpha + 1e-6)
    gamma_power = gamma
    theta_alpha = theta / (alpha + 1e-6)
    stress_index = (beta + gamma) / (alpha + 1e-6)
    engagement_index = beta / (alpha + theta + 1e-6)

    is_focus = int(beta_alpha > 1.2)
    is_memory = int(gamma_power > gamma_power)
    is_relaxed = int(theta_alpha < 1.0)
    is_stressed = int(stress_index > 1.3)
    is_engaged = int(engagement_index > 1.0)

    score = is_focus + is_memory + is_relaxed + is_stressed + is_engaged

    d = {
        "delta": delta,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "beta_alpha": beta_alpha,
        "gamma_power": gamma_power,
        "theta_alpha": theta_alpha,
        "stress_index": stress_index,
        "engagement_index": engagement_index,
        "is_focus": is_focus,
        "is_memory": is_memory,
        "is_relaxed": is_relaxed,
        "is_stressed": is_stressed,
        "is_engaged": is_engaged,
        "score": score
    }
    return pd.DataFrame([d])


def predict_and_prob_from_raw(raw):
    df = compute_features_array(raw)
    try:
        prob = model.predict_proba(df)[:, 1][0]
    except Exception:
        pred = model.predict(df)[0]
        prob = float(pred)
    pred_label = model.predict(df)[0]
    score_val = df["score"].iloc[0]
    return ("ON" if pred_label == 1 else "OFF", float(prob), float(score_val))


# ---------------------------
# GUI & Layout
# ---------------------------
root = tk.Tk()
root.title("Brainwave Bulb — Live BCI Dashboard")
root.geometry("1000x600")
root.configure(bg="#111111")

left = tk.Frame(root, bg="#111111")
left.pack(side="left", fill="y", padx=12, pady=12)

mid = tk.Frame(root, bg="#111111")
mid.pack(side="left", fill="y", padx=12, pady=12)

right = tk.Frame(root, bg="#111111")
right.pack(side="right", fill="both", expand=True, padx=12, pady=12)

# ---------------------------
# Bulb Images
# ---------------------------
try:
    on_img = Image.open("bulb_on.png").resize((160, 160))
    off_img = Image.open("bulb_off.png").resize((160, 160))
except:
    from PIL import Image as PILImage
    on_img = PILImage.new("RGBA", (160,160), "#FFD700")
    off_img = PILImage.new("RGBA", (160,160), "#444444")

bulb_on = ImageTk.PhotoImage(on_img)
bulb_off = ImageTk.PhotoImage(off_img)

bulb_label = tk.Label(left, image=bulb_off, bg="#111111")
bulb_label.pack(pady=10)

state_label = tk.Label(left, text="State: IDLE", fg="white", bg="#111111", font=("Helvetica", 14, "bold"))
state_label.pack(pady=6)

conf_label = tk.Label(left, text="Confidence: ---", fg="white", bg="#111111", font=("Arial", 12))
conf_label.pack(pady=6)

score_label = tk.Label(left, text="Activation score: ---", fg="white", bg="#111111", font=("Arial", 12))
score_label.pack(pady=6)


# ---------------------------
# Gradient Focus Meter (Vertical)
# ---------------------------
METER_WIDTH = 48
METER_HEIGHT = 240
meter_canvas = tk.Canvas(mid, width=METER_WIDTH, height=METER_HEIGHT, bg="#0b0b0b", highlightthickness=0)
meter_canvas.pack(pady=10)

meter_canvas.create_rectangle(2, 2, METER_WIDTH-2, METER_HEIGHT-2, outline="#333", width=2)
bg_rect = meter_canvas.create_rectangle(4, 4, METER_WIDTH-4, METER_HEIGHT-4, fill="#111111", outline="")
fill_rect = meter_canvas.create_rectangle(4, METER_HEIGHT-4, METER_WIDTH-4, METER_HEIGHT-4, fill="#00ff00", outline="")
meter_pct_text = meter_canvas.create_text(METER_WIDTH//2, METER_HEIGHT + 16, text="0%", fill="white", font=("Arial", 11))

def interp_color(val):
    if val <= 50:
        t = val / 50
        r = int(0 + (255-0)*t)
        g = 200
    else:
        t = (val-50)/50
        r = 255
        g = int(200 - 200*t)
    return f"#{r:02x}{g:02x}00"


current_fill_pct = 0.0
target_fill_pct = 0.0

def animate_meter_step():
    global current_fill_pct
    diff = target_fill_pct - current_fill_pct
    step = max(0.6, abs(diff)*0.15)
    current_fill_pct += step if diff > 0 else -step
    if abs(diff) < 0.5:
        current_fill_pct = target_fill_pct

    fill_h = int((current_fill_pct/100) * (METER_HEIGHT - 8))
    y_top = (METER_HEIGHT-4) - fill_h
    meter_canvas.coords(fill_rect, 4, y_top, METER_WIDTH-4, METER_HEIGHT-4)
    meter_canvas.itemconfig(fill_rect, fill=interp_color(current_fill_pct))
    meter_canvas.itemconfig(meter_pct_text, text=f"{int(current_fill_pct)}%")

    if abs(target_fill_pct - current_fill_pct) > 0.5:
        root.after(40, animate_meter_step)


# ---------------------------
# Control Buttons
# ---------------------------
btn_frame = tk.Frame(left, bg="#111111")
btn_frame.pack(pady=6)

start_btn = ttk.Button(btn_frame, text="Start", width=12)
stop_btn = ttk.Button(btn_frame, text="Stop", width=12)
single_btn = ttk.Button(btn_frame, text="Simulate Once", width=12)

start_btn.grid(row=0, column=0, padx=6, pady=6)
stop_btn.grid(row=0, column=1, padx=6, pady=6)
single_btn.grid(row=1, column=0, columnspan=2, pady=6)


# ---------------------------
# Live Graph (Right)
# ---------------------------
fig = Figure(figsize=(6,4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Activation Probability (live)", color="white")
ax.set_ylim(0,1)
ax.set_xlim(0,100)
ax.set_facecolor("#0d1117")
ax.tick_params(colors="white")

canvas = FigureCanvasTkAgg(fig, master=right)
canvas.get_tk_widget().pack(fill="both", expand=True)

MAX_POINTS = 100
prob_buf = deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS)
x_buf = deque(list(range(-MAX_POINTS+1,1)), maxlen=MAX_POINTS)

# Prediction smoothing buffer
prob_smooth_buf = deque([], maxlen=5)


# ---------------------------
# Plot update
# ---------------------------
def update_plot():
    ax.clear()
    ax.set_title("Activation Probability (live)", color="white")
    ax.set_ylim(0,1)
    ax.set_xlim(max(0, x_buf[-1]-MAX_POINTS+1), x_buf[-1]+1)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.plot(list(x_buf), list(prob_buf), color="#00ff99")
    canvas.draw_idle()


# ---------------------------
# Simulation Step
# ---------------------------
def step_simulation(single=False):
    global target_fill_pct

    raw = np.array([
        np.random.uniform(0.1, 5.0),
        np.random.uniform(0.1, 5.0),
        np.random.uniform(0.1, 5.0),
        np.random.uniform(0.1, 6.0),
        np.random.uniform(0.01, 3.0),
        np.random.uniform(-2.0, 5.0),
        np.random.uniform(0.0, 3.0),
        np.random.uniform(0.0, 5.0),
        np.random.uniform(-5.0, 0.0)
    ])

    raw_label, raw_prob, score = predict_and_prob_from_raw(raw)

    # Smoothing
    prob_smooth_buf.append(raw_prob)
    prob = sum(prob_smooth_buf) / len(prob_smooth_buf)
    label = "ON" if prob > 0.5 else "OFF"

    x_buf.append(x_buf[-1]+1)
    prob_buf.append(prob)
    update_plot()

    # Focus meter target
    target_fill_pct = (score / 5) * 100
    animate_meter_step()

    root.after(0, lambda: bulb_label.configure(image=bulb_on if label == "ON" else bulb_off))
    root.after(0, lambda: state_label.configure(text=f"State: {label}"))
    root.after(0, lambda: conf_label.configure(text=f"Confidence: {prob:.2f}"))
    root.after(0, lambda: score_label.configure(text=f"Activation score: {score:.1f}"))

    if single:
        return


# ---------------------------
# Thread Loop
# ---------------------------
stop_event = threading.Event()
worker = None

def worker_loop(interval=0.7):
    while not stop_event.is_set():
        step_simulation()
        time.sleep(interval)

def start_sim():
    global worker
    if worker and worker.is_alive(): return
    stop_event.clear()
    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()
    state_label.configure(text="State: RUNNING")

def stop_sim():
    stop_event.set()
    state_label.configure(text="State: STOPPED")

def single_step():
    step_simulation(single=True)

start_btn.config(command=start_sim)
stop_btn.config(command=stop_sim)
single_btn.config(command=single_step)


# ---------------------------
# Close event
# ---------------------------
def on_close():
    stop_sim()
    time.sleep(0.1)
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
