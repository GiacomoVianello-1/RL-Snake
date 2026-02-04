import torch
import environments_fully_observable 
import environments_partially_observable
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pandas as pd
import os
import pickle


# Check for GPU availability (CUDA for Nvidia, MPS for Apple Silicon)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU Found: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS Found: Apple Silicon Acceleration enabled.")
else:
    device = torch.device("cpu")
    print("No GPU Found. Using CPU: this may be slow, consider reducing the number of parallel environments and training steps.")

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

# Full Environment wrapper
def make_env(n_boards, board_size):
    return environments_fully_observable.OriginalSnakeEnvironment(n_boards,board_size)

def get_env(n=1000):
    size = 7
    return make_env(n, size)

# Safety mask fuction: checks for unsafe moves for all boards
def get_safety_mask(env):
    masks = np.zeros((env.n_boards, 4), dtype=np.float32)
    heads = np.argwhere(env.boards == env.HEAD)
    heads = heads[heads[:, 0].argsort()]

    moves = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}

    for b, hx, hy in heads:
        for a, (dx, dy) in moves.items():
            nx, ny = hx+dx, hy+dy
            if env.boards[b, nx, ny] == env.WALL:
                masks[b, a] = 1.0   # unsafe for wall
    return torch.tensor(masks, device=device)

# Plotting Stuff
# Plot training curves (for comparison)
def plot_training_curves(
    algo_name,
    reward_hist,
    fruits_hist,
    wall_hist,
    baseline_reward,
    baseline_fruits,
    baseline_wall,
    save_dir="results",
    window=200,
    save=False
):
    """
    Generate and save training curves (reward, fruits, wall collisions)
    for any RL algorithm: PPO, A2C, DDQN.
    """

    # Create directory if missing
    os.makedirs(save_dir, exist_ok=True)

    # Moving average
    kernel = np.ones(window) / window
    reward_moving = np.convolve(reward_hist, kernel, mode='valid')
    fruits_moving = np.convolve(fruits_hist, kernel, mode='valid')
    wall_moving = np.convolve(wall_hist, kernel, mode='valid')

    # Plot
    plt.figure(figsize=(15, 4))
    plt.rcParams.update({'font.size': 12})

    # Reward
    plt.subplot(1, 3, 1)
    plt.plot(reward_hist, color='lightblue', alpha=0.5, linewidth=1, label="Raw Reward")
    plt.plot(reward_moving, color='blue', linewidth=2.5, label="Moving Average")
    plt.axhline(y=baseline_reward, color='black', linestyle='--', linewidth=2, label="Baseline")
    plt.title(f"{algo_name.upper()} – Reward per Step")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.legend(fontsize='x-small')
    plt.grid(True, alpha=0.3)

    # Fruits
    plt.subplot(1, 3, 2)
    plt.plot(fruits_hist, color='lightgreen', alpha=0.5, linewidth=1, label="Raw Fruits")
    plt.plot(fruits_moving, color='green', linewidth=2.5, label="Moving Average")
    plt.axhline(y=baseline_fruits, color='black', linestyle='--', linewidth=2, label="Baseline")
    plt.title(f"{algo_name.upper()} – Fruits per Step")
    plt.xlabel("Training Step")
    plt.ylabel("Fruits")
    plt.legend(fontsize='x-small')
    plt.grid(True, alpha=0.3)

    # Wall collisions
    plt.subplot(1, 3, 3)
    plt.plot(wall_hist, color='lightcoral', alpha=0.5, linewidth=1, label="Raw Collisions")
    plt.plot(wall_moving, color='red', linewidth=2.5, label="Moving Average")
    plt.axhline(y=baseline_wall, color='black', linestyle='--', linewidth=2, label="Baseline")
    plt.title(f"{algo_name.upper()} – Wall Collisions per Step")
    plt.xlabel("Training Step")
    plt.ylabel("Wall Collisions")
    plt.legend(fontsize='x-small')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        # Save PDF
        out_path = os.path.join(save_dir, f"{algo_name}_training_curves.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves for {algo_name.upper()} --> {out_path}")
    plt.show()


def plot_comparison_curves(
    ppo_data,
    a2c_data,
    ddqn_data,
    baseline_reward,
    baseline_fruits,
    baseline_wall,
    window=200,
    save=False,
    save_path="results/comparison_training_curves.pdf"
):
    """
    Plot comparison of PPO, A2C, and DDQN training curves.
    """

    # Unpack histories
    ppo_reward, ppo_fruits, ppo_wall = ppo_data
    a2c_reward, a2c_fruits, a2c_wall = a2c_data
    ddqn_reward, ddqn_fruits, ddqn_wall = ddqn_data

    # Moving average kernel
    kernel = np.ones(window) / window

    def smooth(x):
        return np.convolve(x, kernel, mode='valid')

    # Smoothed curves
    ppo_r, ppo_f, ppo_w = smooth(ppo_reward), smooth(ppo_fruits), smooth(ppo_wall)
    a2c_r, a2c_f, a2c_w = smooth(a2c_reward), smooth(a2c_fruits), smooth(a2c_wall)
    ddqn_r, ddqn_f, ddqn_w = smooth(ddqn_reward), smooth(ddqn_fruits), smooth(ddqn_wall)

    # Plot
    plt.figure(figsize=(18, 5))
    plt.rcParams.update({'font.size': 12})

    # REWARD COMPARISON
    plt.subplot(1, 3, 1)
    plt.plot(a2c_r, label="A2C", linewidth=2)
    plt.plot(ddqn_r, label="DDQN", linewidth=2)
    plt.plot(ppo_r, label="PPO", linewidth=2)
    plt.axhline(baseline_reward, color="black", linestyle="--", linewidth=2, label="Baseline")
    plt.title("Reward per Step (Moving Average)")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # FRUITS COMPARISON
    plt.subplot(1, 3, 2)
    plt.plot(a2c_f, label="A2C", linewidth=2)
    plt.plot(ddqn_f, label="DDQN", linewidth=2)
    plt.plot(ppo_f, label="PPO", linewidth=2)
    plt.axhline(baseline_fruits, color="black", linestyle="--", linewidth=2, label="Baseline")
    plt.title("Fruits per Step (Moving Average)")
    plt.xlabel("Training Step")
    plt.ylabel("Fruits")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # WALL COLLISIONS
    plt.subplot(1, 3, 3)
    plt.plot(a2c_w, label="A2C", linewidth=2)
    plt.plot(ddqn_w, label="DDQN", linewidth=2)
    plt.plot(ppo_w, label="PPO", linewidth=2)
    plt.axhline(baseline_wall, color="black", linestyle="--", linewidth=2, label="Baseline")
    plt.title("Wall Collisions per Step (Moving Average)")
    plt.xlabel("Training Step")
    plt.ylabel("Collisions")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if save:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved comparison curves --> {save_path}")
        plt.close()
    else:
        plt.show()

# Let the agent play the game
def display_game(model, agent_name="Agent", max_steps=100):
    game_env = get_env(n=1)
    state = game_env.to_state()
    frames = []

    frames.append(game_env.boards[0].copy())

    # Put model in eval mode
    if hasattr(model, "net"):      # PPOAgent
        model.net.eval()
    else:                          # A2CNet or DDQNNet
        model.eval()

    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        mask = get_safety_mask(game_env)

        with torch.no_grad():

            if hasattr(model, "net"):   # PPO
                logits, _ = model.net(state_tensor)
                logits = logits + mask * -1.0
                actions = torch.argmax(logits, dim=1)

            else:
                out = model(state_tensor)

                if isinstance(out, tuple):   # A2C
                    logits, _ = out
                    logits = logits + mask * -1.0
                    actions = torch.argmax(logits, dim=1)

                else:                        # DDQN
                    q_values = out
                    actions = torch.argmax(q_values, dim=1)

        actions_np = actions.cpu().numpy().reshape(-1, 1)

        game_env.move(actions_np)
        state = game_env.to_state()

        frames.append(game_env.boards[0].copy())

    # 1. STATIC SNAPSHOTS EVERY 10 STEPS
    snapshot_steps = [0, 10, 20, 30, 40]
    snapshot_steps = [s for s in snapshot_steps if s < len(frames)]

    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(15, 3))

    for ax, step in zip(axes, snapshot_steps):
        ax.imshow(frames[step], origin='lower', cmap='viridis', vmin=0, vmax=4)
        ax.set_title(f"{agent_name} – Step {step}")
        ax.axis('off')

    plt.show()

    # 2. ANIMATION
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.axis('off')

    img = ax.imshow(frames[0], origin='lower', cmap='viridis', vmin=0, vmax=4)
    ax.set_title(f"Snake Agent Replay – {agent_name}")

    def update(frame):
        img.set_array(frame)
        return [img]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=100,
        blit=True
    )

    plt.show()

# UTILTY FUNCTION TO SAVE TRAINING RESULTS
def save_training_results(
    algo_name,
    model,
    reward_hist,
    fruits_hist,
    deaths_hist,
    extra_metrics=None,
    params=None,
    save_dir="results",
    save=True
):
    """
    Save model weights and training data for any RL algorithm.
    """

    # If saving is disabled, exit gracefully
    if not save:
        print(f"[INFO] Saving disabled --> skipping save for {algo_name}")
        return

    # Create directory
    algo_dir = os.path.join(save_dir, algo_name)
    os.makedirs(algo_dir, exist_ok=True)

    # Detect the actual PyTorch model inside the agent
    if hasattr(model, "state_dict"):
        torch_model = model
    elif hasattr(model, "net"):
        torch_model = model.net
    else:
        raise ValueError(f"Cannot find a PyTorch model inside {algo_name} agent")

    # Save weights
    torch.save(torch_model.state_dict(), os.path.join(algo_dir, f"{algo_name}_weights.pt"))

    # Save training data
    data = {
        "reward_hist": reward_hist,
        "fruits_hist": fruits_hist,
        "deaths_hist": deaths_hist,
        "params": params or {},
    }

    if extra_metrics is not None:
        data.update(extra_metrics)

    with open(os.path.join(algo_dir, f"{algo_name}_training.pkl"), "wb") as f:
        pickle.dump(data, f)

    print(f"Saved {algo_name} results in {algo_dir}")
