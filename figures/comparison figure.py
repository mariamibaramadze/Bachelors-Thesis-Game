import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")     # figures/ -> ../results/
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "generated")          # figures/generated/
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROLLING_WINDOW = 25
COLORS = {"EASY": "#2D8A3E", "MEDIUM": "#1A5FA8", "HARD": "#CC2222"}
LABELS = {"EASY": "Easy", "MEDIUM": "Medium", "HARD": "Hard"}

data = {}
for key in ["EASY", "MEDIUM", "HARD"]:
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"{key}_results.csv"))
    df.columns = [c.strip() for c in df.columns]
    df["RollingReward"] = df["Reward"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df["RollingSuccess"] = df["Success"].rolling(ROLLING_WINDOW, min_periods=1).mean() * 100
    data[key] = df

# ---------- Figure 7: combined learning curves (rolling reward only, all 3 levels) ----------
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
for key in ["EASY", "MEDIUM", "HARD"]:
    df = data[key]
    ax.plot(df["Episode"], df["RollingReward"], color=COLORS[key], linewidth=2.2, label=LABELS[key])
ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
ax.set_xlabel("Episode")
ax.set_ylabel("Rolling average reward (window = 25)")
ax.set_title("Comparison of learning curves across environments")
ax.legend(loc="lower right", frameon=False, fontsize=9, title="Level")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 500)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "figure_7_cross_level_learning_curves.png"), facecolor="white")
plt.close(fig)
print(f"Saved {os.path.join(OUTPUT_DIR, 'figure_7_cross_level_learning_curves.png')}")

# ---------- Figure 8: success rate comparison (rolling success % over episodes) ----------
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
for key in ["EASY", "MEDIUM", "HARD"]:
    df = data[key]
    ax.plot(df["Episode"], df["RollingSuccess"], color=COLORS[key], linewidth=2.2, label=LABELS[key])
ax.set_xlabel("Episode")
ax.set_ylabel("Rolling success rate (%, window = 25)")
ax.set_title("Success rate comparison across environments")
ax.legend(loc="lower right", frameon=False, fontsize=9, title="Level")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 500)
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "figure_8_success_rate_comparison.png"), facecolor="white")
plt.close(fig)
print(f"Saved {os.path.join(OUTPUT_DIR, 'figure_8_success_rate_comparison.png')}")

# ---------- Figure 9: average steps to success comparison (bar chart, final summary) ----------
summary = {}
for key in ["EASY", "MEDIUM", "HARD"]:
    s = pd.read_csv(os.path.join(RESULTS_DIR, f"{key}_summary.csv"))
    s.columns = [c.strip() for c in s.columns]
    summary[key] = s.iloc[0]

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
levels = ["EASY", "MEDIUM", "HARD"]
vals = [summary[k]["AvgStepsToSuccess"] for k in levels]
bar_colors = [COLORS[k] for k in levels]
bars = ax.bar([LABELS[k] for k in levels], vals, color=bar_colors, width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)
ax.set_ylabel("Average steps to success")
ax.set_title("Average steps to success across environments")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "figure_9_avg_steps_comparison.png"), facecolor="white")
plt.close(fig)
print(f"Saved {os.path.join(OUTPUT_DIR, 'figure_9_avg_steps_comparison.png')}")