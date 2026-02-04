# Reinforcement Learning -- Snake Game

This repository contains the implementation of multiple Deep Reinforcement Learning (DRL) agents trained to play the classic game "Snake".

This project was developed for the **Reinforcement Learning course (2025/2026)** at the **University of Padua**.

## 📋 Project Overview
The goal of this project is to develop an autonomous agent capable of maximizing the score (fruits eaten) in a fixed-grid Snake environment while avoiding collisions with walls and its own body.

We implemented and compared three distinct DRL algorithms against a Heuristic Baseline:

1.  **PPO (Proximal Policy Optimization)**: An on-policy gradient method that uses a clipped objective function to ensure stable training.
2.  **A2C (Advantage Actor-Critic)**: A synchronous, deterministic variant of A3C that utilizes an advantage function to reduce variance.
3.  **DDQN (Double Deep Q-Network)**: An off-policy value-based method that addresses the overestimation bias found in standard DQN.

## ⚙️ Installation & Requirements

Ensure you have Python installed (Python 3.10+ recommended). Install the required dependencies:

```
pip install numpy torch pandas tqdm matplotlib
```

*Note: A GPU (CUDA/MPS) is recommended for training, but the evaluation script runs efficiently on CPU.*

## 📂 Repository Structure
The codebase is modularized to ensure readability and reproducibility:

* `evaluate.py`: **Main entry point.** Loads pre-trained weights and evaluates all agents against the baseline.
* `train.py`: Script to train all agents from scratch and save weights/logs to the `results/` folder.
* `models.py`: PyTorch definitions of the Neural Network architectures (CNNs).
* `agents.py`: Implementation of agent logic (PPO, A2C, DDQN) and memory buffers.
* `utils.py`: Helper functions for environment setup, seeding, and device management.
* `environments_fully_observable.py`: The Snake environment logic (provided by the course).
* `results/`: Directory containing pre-trained model weights (`.pt`) and training logs (`.pkl`).
* `main.ipynb`: Jupyter Notebook used for initial development, visualization, and plotting.

## 🚀 Usage

1. **Evaluation**
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

2. **Training**
    If you wish to retrain the models from scratch:
    ```
    python train.py
    ```
    This will sequentially train PPO, A2C, and DDQN.
    - **Warning:** Training can take several hours depending on your hardware.
    - Weights and training history can be automatically saved to `results/<algorithm_name>/`.

## 📊 Results
The agents are evaluated based on three metrics:
- **Average Reward:** The mean reward accumulated per step.
- **Fruits (Avg):** The average length of the snake (proxy for game score).
- **Wall Hits (Avg):** Safety metric indicating how often the agent crashes into walls (valid only for non-terminal testing phases).

Full training curves and comparisons can be viewed either by running the `main.ipynb` notebook, inspecting the `results/` directory, or by running the evaluation script as mentioned above.