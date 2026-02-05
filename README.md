# Deep Reinforcement Learning for Snake

This repository contains a PyTorch implementation of three distinct Deep Reinforcement Learning (DRL) algorithms -- **Proximal Policy Optimization (PPO)**, **Advantage Actor-Critic (A2C)**, and **Double Deep Q-Network (DDQN)** -- applied to the classic game of Snake.

The project investigates the efficacy of these paradigms in a discrete, grid-based navigation task requiring long-term planning. A key contribution of this work is the implementation of a **massively vectorized environment** capable of simulating thousands of parallel games on a single GPU, ensuring high-throughput training and stable convergence.

## 📌 Project Overview
The objective is to train an autonomous agent to maximize fruit consumption within a fixed $7 \times 7$ grid while avoiding (if possible) collisions with walls and its own growing body. The problem is modeled as a **continuing MDP**, where the agent must balance immediate rewards (fruit collection) with long-term survival strategies (space management).

**Key Features**
- **Vectorized Environments:** Custom-built Python environment using numpy to simulate $N$ parallel boards simultaneously, significantly accelerating data collection.
- **Unified Architecture:** All agents share an identical 3-layer CNN backbone to ensure that performance differences are attributable solely to the algorithmic logic.
- **Safety Masking:** Implementation of an inductive bias that filters out immediate wall collisions during action sampling, improving sample efficiency
- **Algorithms:** implementations of PPO (Clipped Surrogate), A2C (Synchronous), and DDQN (Off-policy with Replay Buffer).

## 🧠 Neural Architecture

To facilitate a fair comparison, all agents utilize a standardized **Convolutional Neural Network (CNN)** architecture designed to extract spatial features from the grid state $s_t \in \mathbb{R}^{7 \times 7 \times 4}$.

The network processes the input (one-hot encoded channels: *Head*, *Body*, *Fruit*, *Empty*) through the following stages:
1. **Feature Extraction:** 
    - `Conv2d`: 32 filters, $3\times3$ kernel, ReLU
    - `Conv2d`: 64 filters, $3\times3$ kernel, ReLU
    - `Conv2d`: 128 filters, $3\times3$ kernel, ReLU
2. **Latent Representation:**
    - Flattening followed by a Dense Layer (128 units) with ReLU activation.
3. **Output Heads (Algorithm Dependent):**
    - **PPO/A2C:** Dual heads for **Policy** ($\pi(a|s)$) and **Value** ($V(s)$).
    - **DDQN:** Single head for **Q-Values** ($Q(s, a)$).

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
├── agents.py                            # Implementations of PPOAgent (updates, GAE) and buffers
├── environments_fully_observable.py     # Vectorized Snake environment logic
├── environments_partially_observable.py # logic for PO (Partial Observability)
├── evaluate.py                          # Script for evaluating trained models
├── main.ipynb                           # Jupyter Notebook for experimentation and visualization
├── models.py                            # PyTorch Neural Network definitions (PPOActorCritic, A2CNet, DDQNNet)
├── train_a2c.py                         # Training loop for A2C
├── train_ddqn.py                        # Training loop for Double DQN
├── train_ppo.py                         # Training loop for PPO
├── utils.py                             # Utilities: seeding, device management, saving/loading
└── results/                             # Directory storing training logs and model weights
```

## 🚀 Usage

**Prerequisites**
- Python 3.8+
- PyTorch (CUDA recommended for performance)
- NumPy
- Tqdm
- Matplotlib

**Training an Agent**
If you wish to retrain the models from scratch:
```
python train_AGENT.py
```
where AGENT=`ddqn`,`a2c`, or `ppo`.
- **Warning:** Training can take several hours depending on your hardware.
- *Configuration:* You can modify hyperparameters such as `NUM_BOARDS` (default 1000) and `ITERATIONS` (default 20M) directly in the script.
- Weights and training history can be automatically saved to `results/<algorithm_name>/`.

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
The agents are evaluated based on three metrics:
- **Average Reward:** The mean reward accumulated per step.
- **Fruits (Avg):** The average length of the snake (proxy for game score).
- **Wall Hits (Avg):** Safety metric indicating how often the agent crashes into walls (valid only for non-terminal testing phases).

Our experiments demonstrate that PPO significantly outperforms both A2C and DDQN in terms of stability and final score.
- **PPO:** Converges to a near-optimal policy, effectively avoiding self-collisions even at high snake lengths, though it does not win the game so often.
- **A2C:** Shows good convergence speed but exhibits higher variance and occasional instability in long-horizon episodes.
- **DDQN:** Struggles with the sparse reward structure of the late-game phase, often failing to learn safe turning maneuvers despite the implementation of Double Q-learning.

Full training curves and comparisons can be viewed either by running the `main.ipynb` notebook, inspecting the `results/` directory, or by running the evaluation script as mentioned above.