# 🎮 Q-Bunny: Reinforcement Learning Game Agent

## 📌 Project Overview

This project is a reinforcement learning (RL) based game where an AI agent learns to play a custom Pygame environment.
The goal of the agent is to collect rewards (carrots) while avoiding obstacles (wolf) across different difficulty levels.

The project was developed as part of a Bachelor's Thesis focusing on **reinforcement learning performance in environments of increasing complexity**.

---

## 🧠 Technologies Used

* Python 3.x
* Pygame (game environment and rendering)
* NumPy (Q-table and numerical operations)
* Reinforcement Learning (tabular Q-learning)
* Pandas and Matplotlib (used only by the scripts in `figures/`, to analyze results and generate charts — not required to run the game itself)

---

## 🎯 Game Description

The environment includes:

* 🐰 Bunny (agent controlled by AI)
* 🥕 Carrot (reward / goal)
* 🐺 Wolf (penalty / danger, present in Medium and Hard only)

### Difficulty Levels

* **Easy** — open 5×5 grid, no obstacles or threats, carrot moves once per episode
* **Medium** — 8×8 grid with static walls and a wolf that follows a fixed patrol route, carrot moves every step
* **Hard** — 11×11 maze layout with a wolf that actively chases the agent, carrot moves every step

Each level adds a specific new source of difficulty rather than simply enlarging the grid, so that the effect of each change can be compared individually.

---

## 📊 Training Results

Each level is trained for 500 episodes. Two files are saved per level into the `results/` folder:

* `EASY_results.csv`, `MEDIUM_results.csv`, `HARD_results.csv` — one row per episode, recording episode number, total reward, whether the episode ended in success, and how many steps it took.
* `EASY_summary.csv`, `MEDIUM_summary.csv`, `HARD_summary.csv` — a single summary row per level, recording overall success rate, average reward, average steps to success, best reward achieved, and total training time.

---

## ⚙️ How It Works

1. The agent starts with no knowledge of the environment.
2. It explores using epsilon-greedy action selection.
3. It receives rewards or penalties based on its actions.
4. It updates its Q-table using the Q-learning (Bellman) update rule.
5. Epsilon decays over time, shifting the agent from exploration to exploitation as training progresses.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/mariamibaramadze/Bachelors-Thesis-Game.git
```

### 2. Install dependencies

To run the game itself:

```bash
pip install pygame numpy
```

To also run the figure-generation scripts in `figures/`:

```bash
pip install pandas matplotlib
```

### 3. Run the game

```bash
python Q-Bunny.py
```

This trains the agent on all three levels in sequence, rendering live, and saves the resulting CSV files into `results/`.

### 4. Generate result figures (optional)

The scripts in `figures/` read from `results/` and write rendered charts into `figures/generated/`. Run them from inside the `figures/` folder:

```bash
cd figures
python "learning curve.py"
python "comparison figure.py"
```

The two environment-diagram scripts (`figure 1.py` for Easy, `figure 2.py` for Medium/Hard) read sprite images from `assets/` and can be run the same way.

---

## 📁 Project Structure

```
BachelorsThesisGame/
│
├── Q-Bunny.py
├── README.md
├── .gitignore
│
├── assets/
│   ├── bunny.png
│   ├── carrot.PNG
│   └── wolf.PNG
│
├── results/
│   ├── EASY_results.csv
│   ├── EASY_summary.csv
│   ├── MEDIUM_results.csv
│   ├── MEDIUM_summary.csv
│   ├── HARD_results.csv
│   └── HARD_summary.csv
│
└── figures/
    ├── figure 1.py            # Easy level environment diagram
    ├── figure 2.py            # Medium/Hard level environment diagram
    ├── learning curve.py      # Per-level learning curves
    ├── comparison figure.py   # Cross-level comparison charts
    └── generated/
        ├── Figure 1.png
        ├── figure 2.png
        ├── figure_4_easy_learning_curve.png
        ├── figure_5_medium_learning_curve.png
        ├── figure_6_hard_learning_curve.png
        ├── figure_7_cross_level_learning_curves.png
        ├── figure_8_success_rate_comparison.png
        └── figure_9_avg_steps_comparison.png
```

---

## 📌 Author

Bachelor's Thesis Project
Student: Mariam Baramadze

---

## 📖 Notes

* This project investigates how the performance of a tabular Q-learning agent changes as environmental complexity increases, using three progressively harder versions of the same game.
* It is designed for educational and research purposes as part of a Bachelor's Thesis in Computer Science.