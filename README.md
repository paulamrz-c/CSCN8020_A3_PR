# 🕹️ CSCN8020 - Assignment 3  

Welcome to my repository for **Assignment 3 of CSCN8020 – Reinforcement Learning**.  

This project implements a **Deep Q-Network (DQN)** agent to play the **Atari Pong** environment using **OpenAI Gymnasium**.  
The goal was to train the agent to improve its performance by learning from experience and adjusting Q-values through a Convolutional Neural Network (CNN).  

---

## 📓 Repository Content  

- **Main Notebook**  
  `demo_assignment_3.ipynb`  
  Contains the training experiments, plots, and analysis of different hyperparameter combinations.  

- **Python Scripts**
  - `main.py` → Main training loop that manages episodes, memory, and TensorBoard logging.  
  - `agent.py` → Implementation of the DQN agent and CNN architecture.  
  - `assignment3_utils.py` → Helper classes for preprocessing, replay memory, and frame stacking.  

- **Results Folder**  
  The folder `runs/` contains all TensorBoard logs for experiments with different batch sizes and target update rates.  

---

## ⚙️ Hyperparameters  

| Hyperparameter | Value | Description |
|----------------|--------|-------------|
| **Discount factor (γ)** | `0.95` | Balances short- and long-term rewards. |
| **Exploration rate (ε initial)** | `1.0` | Starts with full exploration using random actions. |
| **Exploration decay (δ)** | `0.995` | Gradually reduces exploration after each episode. |
| **Minimum exploration (ε_min)** | `0.05` | Keeps small exploration at the end of training. |
| **Mini-batch size** | `8` *(default)* / `16` *(in experiments)* | Number of samples used for each training step. |
| **Target update rate** | `10` *(default)* / `3` *(in experiments)* | Frequency of updating target network weights. |
| **Learning rate (α)** | `0.00025` | Learning rate for the Adam optimizer. |
| **Replay memory size** | `10,000` | Maximum number of stored experiences. |
| **Training frequency** | Every 25 steps | Network updates occur after a set number of steps. |
| **Input frames (stack)** | `4` | Number of consecutive frames stacked as input. |

---

## 📈 Metrics  

| Metric | Description |
|---------|-------------|
| **Episode Reward** | Total game score per episode. |
| **Average Reward (last 5 episodes)** | Measures recent training stability and progress. |
| **Loss** | Shows the difference between predicted and target Q-values. |
| **Epsilon (ε)** | Current exploration rate, decreases gradually. |

---

## 🧠 Experiments  

Experiments were conducted by varying **batch size** and **update rate** across 50 and 100 episodes.  

| # | Episodes | Batch | Update | Avg. Reward | Notes |
|:-:|:---------:|:------:|:--------:|:-------------:|:------|
| 1 | 50 | 8 | 10 | ≈ -20 | Stable but with no improvement. |
| 2 | 50 | 16 | 10 | ≈ -19.4 | More stable, smoother loss. |
| 3 | 50 | 8 | 3 | ≈ -19.4 | Slight early improvement. |
| 4 | 50 | 16 | 3 | ≈ -19.4 | Most consistent configuration. |
| 5 | 100 | 8 | 10 | ≈ -19.4 | Still no major progress. |
| 6 | 100 | 16 | 10 | ≈ -19.6 | Stable and smooth curve. |
| 7 | 100 | 8 | 3 | ≈ -20 | Slight early improvement, then stagnation. |
| 8 | 100 | 16 | 3 | ≈ -20 | Stable with small peaks, consistent learning. |

---

## 🏆 Best Configurations  

| Rank | Configuration | Episodes | Batch | Update | Avg. Reward | Notes |
|------|----------------|-----------|--------|----------|--------------|--------|
| 🥇 **1** | #4 (50 episodes, batch 16, update 3) | 50 | 16 | 3 | ≈ -19.4 | Most stable training, smoother loss. |
| 🥈 **2** | #6 (100 episodes, batch 16, update 10) | 100 | 16 | 10 | ≈ -19.6 | Consistent training and smoother convergence. |

> The **batch size of 16** consistently provided better and more stable results.  
> Updating the target network less frequently (every 3–10 episodes) helped prevent unstable Q-value jumps.

---

## 🚀 How to Run  

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.env\Scriptsctivate

# or Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the training
python main.py
```

To visualize results in TensorBoard:
```bash
tensorboard --logdir runs
```

---

## 🧩 Key Findings  

- Increasing the number of episodes from 50 to 100 made training more stable.  
- Larger batch sizes (16) reduced variance and improved reward consistency.  
- Frequent updates (update rate = 3) helped early improvements but became less stable in long runs.  
- The model with **batch size 16** and **update rate 3** achieved the best overall balance.  

---

## 🏁 Summary  

> After analyzing multiple configurations, the **DQN agent with batch size 16 and update rate 3** produced the most stable learning curve.  
> Although the reward values remain negative, this setup shows consistent convergence, indicating the model is learning gradually and would likely improve with longer training.  

---

## 📎 Notes  

Because the submission platform allows only PDF files, the **GitHub repository link** is included in the report to access the full code, results, and trained models.

---
