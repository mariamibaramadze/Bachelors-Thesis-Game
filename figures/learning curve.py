import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")     # figures/ -> ../results/
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "generated")          # figures/generated/
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEVELS = [
    ("EASY",   "figure_4_easy_learning_curve.png",   "Easy"),
    ("MEDIUM", "figure_5_medium_learning_curve.png",  "Medium"),
    ("HARD",   "figure_6_hard_learning_curve.png",    "Hard"),
]

ROLLING_WINDOW = 25

for level_key, outfile, label in LEVELS:
    csv_path = os.path.join(RESULTS_DIR, f"{level_key}_results.csv")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    rolling = df["Reward"].rolling(ROLLING_WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

    # Raw episode rewards (faint scatter)
    ax.scatter(df["Episode"], df["Reward"], s=6, color="#9DB8C9",
               alpha=0.45, linewidths=0, label="Episode reward (raw)", zorder=2)

    # Rolling average (bold line)
    ax.plot(df["Episode"], rolling, color="#1A5FA8", linewidth=2.2,
            label=f"Rolling average (window = {ROLLING_WINDOW})", zorder=3)

    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"Learning curve — {label} level")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, df["Episode"].max())

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, outfile)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")