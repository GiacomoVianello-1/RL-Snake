# Deep Reinforcement Learning for Snake

This repository contains a PyTorch implementation of distinct Deep Reinforcement Learning (DRL) algorithms -- **Proximal Policy Optimization (PPO)**, **Advantage Actor-Critic (A2C)**, and **Double Deep Q-Network (DDQN)** -- applied to the classic game of Snake.

The project investigates the efficacy of these paradigms in a discrete, grid-based navigation task requiring long-term planning. A key contribution of this work is the implementation of a **massively vectorized environment** capable of simulating thousands of parallel games on a single GPU.

Additionally, this project explores the **impact of Partial Observability**, comparing standard "allocentric" agents (global view) against "egocentric" agents restricted to a $3 \times 3$ window, highlighting the trade-off between local survival and global planning efficiency.

## ✅ Evaluation results
We prepared 2 scripts for evaluating our 3 agents.
1. To test the RL models vs. the baseline in the full observability case, run
    ```
    python evaluate.py
    ```
2. To test the PPO model (specifically pre-trained for the partially observable environment), run
    ```
    python evaluate_partial.py
    ```
Both scripts will generate a comparison table to assess the performance based on the metrics discussed below.


## 📌 Project Overview
The objective is to train an autonomous agent to maximize fruit consumption within a fixed $7 \times 7$ grid while avoiding collisions with walls and its own growing body. The problem is modeled as a **continuing MDP**, where the agent must balance immediate rewards (fruit collection) with long-term survival strategies (space management).

**Key Features**
- **Vectorized Environments:** Custom-built Python environment using numpy to simulate $N$ parallel boards simultaneously, significantly accelerating data collection.
- **Unified Architecture:** All agents share an identical 3-layer CNN backbone to ensure that performance differences are attributable solely to the algorithmic logic.
- **Safety Masking:** Implementation of an inductive bias that filters out immediate wall collisions during action sampling, improving sample efficiency.
- **Partial Observability Analysis:** Dedicated experiments testing agent robustness under severe visual constraints ($3 \times 3$ egocentric view).
- **Algorithms:** Implementations of PPO (Clipped Surrogate), A2C (Synchronous), and DDQN (Off-policy with Replay Buffer).

## 🧠 Neural Architecture

To facilitate a fair comparison, all agents utilize a standardized **Convolutional Neural Network (CNN)** architecture designed to extract spatial features from the grid state.

The network processes the input (one-hot encoded channels: *Head*, *Body*, *Fruit*, *Empty*) through the following stages:
1. **Feature Extraction:** - `Conv2d`: 32 filters, $3\times3$ kernel, ReLU
    - `Conv2d`: 64 filters, $3\times3$ kernel, ReLU
    - `Conv2d`: 128 filters, $3\times3$ kernel, ReLU
2. **Latent Representation:**
    - Flattening followed by a Dense Layer (128 units) with ReLU activation.
3. **Output Heads (Algorithm Dependent):**
    - **PPO/A2C:** Dual heads for **Policy** ($\pi(a|s)$) and **Value** ($V(s)$).
    - **DDQN:** Single head for **Q-Values** ($Q(s, a)$).

*Note: For the partially observable agent, the input dimension adapts to the $3 \times 3$ window size, but the convolutional backbone remains consistent.*

## 🌍 Environment & Reward Function

Environments are defined in `environments_fully_observable.py` and `environments_partially_observable.py`. The reward function $R(s, a, s')$ is shaped to guide the agent toward optimal behavior:

<div align="center">

| **Event** | **Reward** | **Motivation** |
| :---  | :---:  | :---:      |
| Win (Board Full) |+10.0     | Strongly incentivize the rare event of perfect completion.  |
| Eat Fruit        | +0.5     | Provide positive reinforcement for the primary objective.   |
| Step             | -0.01    | Discourage loitering and encourage shortest-path navigation.|
| Hit Wall         | -0.1     | Penalty for boundary collisions (mitigated by Safety Mask). |
| Self-Collision   | -0.5     | Severe penalty to enforce structural integrity and prevent self-trapping. |

</div>

## 📂 Repository Structure

```
. 
├── agents.py                               # Implementations of PPOAgent (updates, GAE) and buffers 
├── environments_fully_observable.py        # Vectorized Snake environment (Global 7x7 View) 
├── environments_partially_observable.py    # Vectorized environment for Partial Observability 
├── evaluate.py                             # Script for evaluating trained models (comparative analysis) 
├── main.ipynb                              # Jupyter Notebook for experimentation and visualization 
├── models.py                               # PyTorch Neural Network definitions 
├── train_a2c.py                            # Training loop for A2C 
├── train_ddqn.py                           # Training loop for Double DQN 
├── train_ppo.py                            # Training loop for PPO (Full Observability) 
├── train_ppo_partially.py                  # Training loop for PPO (Partial 3x3 Observability) 
├── utils.py                                # Utilities: seeding, device management, saving/loading 
└── results/                                # Directory storing training logs and model weights
```

## 🚀 Usage

**Prerequisites**
- Python 3.8+
- PyTorch (CUDA recommended for performance)
- NumPy
- Tqdm
- Matplotlib

**Training an Agent**
To train the standard agents on the full environment:
```bash
python train_ppo.py   # Trains PPO
python train_a2c.py   # Trains A2C
python train_ddqn.py  # Trains DDQN
```
- **Warning:** Training can take several hours depending on your hardware.
- *Configuration:* You can modify hyperparameters such as `NUM_BOARDS` (default 1000) and `ITERATIONS` (default 20M) directly in the script.
- Weights and training history can be automatically saved to `results/<algorithm_name>/`.

To train the Partially Observable PPO agent (3x3 view):
```
python train_ppo_partially.py
```

**Evaluation**
To evaluate the pre-trained agents as required by the project specifications, simply run:
```
python evaluate.py
```
This script will:
- Initialize the environment.
- Load the best weights for PPO, A2C, and DDQN from the results/ folder.
- Run the Heuristic Baseline for comparison.
- Run a specified number of evaluation steps for each agent.
- Output a comparison table showing Average Reward, Fruits Eaten, and Wall Collisions.

## 📊 Results Summary
The agents are evaluated on **Mean Reward**, **Fruits Eaten** (proxy for score), and **Wall Hits** (safety metric).

1. **Algorithm Comparison (Full Observability)**
    PPO significantly outperforms A2C and DDQN in stability and final score.
    - **PPO:** Converges to a near-optimal policy, effectively avoiding self-collisions.
    - **A2C:** Good convergence speed but exhibits higher variance in long episodes.
    - **DDQN:** Struggles with sparse rewards in the late game, often failing to learn safe turning maneuvers.

2. **Partial Observability Analysis**
    - We compared the PPO agent in the Fully Observable ($7 \times 7$) setting against the Partially Observable ($3 \times 3$) setting.

**Key Insight:** The "Blind" agent maintains near-perfect survival rates (low wall hits), proving that obstacle avoidance is a local control problem solvable with tunnel vision. However, the drop in fruit consumption ($\approx 38\%$) quantifies the Cost of Blindness: without a global map, the agent cannot plan optimal paths to distant rewards and defaults to a conservative random walk.