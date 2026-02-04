# Reinforcement Learning -- Snake Game
This project implements a full Reinforcement Learning pipeline for the classic Snake game, rewritten entirely in PyTorch and optimized for GPU acceleration. Three different RL algorithms were trained, evaluated, and compared on the same environment:
- **DDQN** -- Double Deep Q‑Network
- **A2C** -- Advantage Actor‑Critic
- **PPO** — Proximal Policy Optimization

The goal is to study how different RL paradigms (value‑based, actor‑critic, policy‑gradient) behave on the same task, under identical conditions.

### Highlights

- The original template was **fully rewritten in PyTorch**, preserving the logic but improving clarity, modularity, and performance.
- Training supports **CUDA acceleration**, reducing training time.
- A unified evaluation pipeline compares all agents against a **greedy baseline policy**.
- A complete visualization suite:
    - training curves (reward, fruits, wall collisions)
    - comparison plots across algorithms
    - real‑time agent gameplay with animation
- A clean and modular codebase suitable for experimentation.

### Important aspects
- The original project template has been completely rewritten (without changing anything in the logic structure) in **PyTorch**.
- We make use of **cuda** acceleration to train the NNs faster (on GPU).
- Three different RL algorithms have been tested and compared:
    - **DDQN**: Double Deep Q-Network.
    - **A2C**: Advantage Actor-Critic.
    - **PPO**: Proximal Policy Optimization.

Source the Python venv:
```
source venv/rl-env/bin/activate
```