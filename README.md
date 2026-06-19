# 🎮 Q-Bunny: Reinforcement Learning Game Agent

## 📌 Project Overview

This project is a reinforcement learning (RL) based game where an AI agent learns to play a custom Pygame environment.
The goal of the agent is to collect rewards (carrots) while avoiding obstacles (wolf) across different difficulty levels.

The project was developed as part of a Bachelor's Thesis focusing on **AI in game environments and reinforcement learning applications**.

---

## 🧠 Technologies Used

* Python 3.x
* Pygame
* NumPy
* Pandas
* Reinforcement Learning (Q-Learning)
* Matplotlib (for result visualization)

---

## 🎯 Game Description

The environment includes:

* 🐰 Bunny (agent controlled by AI)
* 🥕 Carrots (rewards)
* 🐺 Wolf (penalty / danger)

### Difficulty Levels:

* EASY
* MEDIUM
* HARD

Each level increases:

* obstacle difficulty
* learning complexity
* required training episodes

---

## 📊 Training Results

The project saves training results into CSV files:

* EASY_results.csv
* MEDIUM_results.csv
* HARD_results.csv

These logs track agent performance over episodes.

---

## ⚙️ How It Works

1. The agent starts with no knowledge of the environment
2. It explores using Q-learning
3. It receives rewards or penalties
4. It updates its Q-table
5. Over time, it learns an optimal strategy

---

## 🚀 How to Run

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### 2. Install dependencies

```bash
pip install pygame numpy pandas matplotlib
```

### 3. Run the game

```bash
python Q-Bunny.py
```

---

## 📁 Project Structure

```
BachelorsThesisGame/
│
├── Q-Bunny.py
├── bunny.png
├── wolf.PNG
├── carrot.PNG
├── EASY_results.csv
├── MEDIUM_results.csv
├── HARD_results.csv
└── README.md
```

---

## 📌 Author

Bachelor Thesis Project
Student: Mariam Baramadze

---

## 📖 Notes

* This project demonstrates reinforcement learning in a simple game environment.
* It is designed for educational and research purposes.
