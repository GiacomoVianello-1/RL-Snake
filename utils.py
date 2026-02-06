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
from models import DDQNNet


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

# ====================
# Environment wrapper
# ====================

# Fully observable

def make_env(n_boards, board_size):
    return environments_fully_observable.OriginalSnakeEnvironment(n_boards,board_size)

def get_env(n=1000):
    size = 7
    return make_env(n, size) 

# Safety mask fuction: checks for unsafe moves for all boards
def get_safety_mask(env):
    '''
    For each parallel board, identifies the current coordinates of the snake's head and calculates 
    the potential next position for each of the four cardinal moves: UP, RIGHT, DOWN, and LEFT. 
    If a projected move targets a cell occupied by a WALL (represented by value 0), 
    the corresponding action is flagged as "unsafe" in a boolean mask.
    '''
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

# === Plotting Stuff ===

# Plot training curves
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
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.ticker import MaxNLocator

    # Create directory if missing
    os.makedirs(save_dir, exist_ok=True)

    # Aesthetics
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.dpi": 150
    })

    # Moving average
    kernel = np.ones(window) / window
    reward_moving = np.convolve(reward_hist, kernel, mode='valid')
    fruits_moving = np.convolve(fruits_hist, kernel, mode='valid')
    wall_moving = np.convolve(wall_hist, kernel, mode='valid')

    # Colors (scientific palette)
    colors = {
        "raw_reward": "#9ecae1",
        "smooth_reward": "#3182bd",
        "raw_fruits": "#a1d99b",
        "smooth_fruits": "#31a354",
        "raw_wall": "#fcbba1",
        "smooth_wall": "#e34a33",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    # --- REWARD ---
    ax = axes[0]
    ax.plot(reward_hist, color=colors["raw_reward"], alpha=0.4, linewidth=1)
    ax.plot(reward_moving, color=colors["smooth_reward"], linewidth=2)
    ax.axhline(baseline_reward, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"{algo_name.upper()} – Reward per Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")

    # --- FRUITS ---
    ax = axes[1]
    ax.plot(fruits_hist, color=colors["raw_fruits"], alpha=0.4, linewidth=1)
    ax.plot(fruits_moving, color=colors["smooth_fruits"], linewidth=2)
    ax.axhline(baseline_fruits, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"{algo_name.upper()} – Fruits per Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fruits")

    # --- WALL COLLISIONS ---
    ax = axes[2]
    ax.plot(wall_hist, color=colors["raw_wall"], alpha=0.4, linewidth=1)
    ax.plot(wall_moving, color=colors["smooth_wall"], linewidth=2)
    ax.axhline(baseline_wall, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"{algo_name.upper()} – Wall Collisions per Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Collisions")

    # avoid overlapping x-ticks
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis='x', labelrotation=0)

    # legend
    handles = [
        plt.Line2D([], [], color=colors["raw_reward"], alpha=0.4, linewidth=2, label="Raw"),
        plt.Line2D([], [], color=colors["smooth_reward"], linewidth=2, label="Moving Average"),
        plt.Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="Baseline")
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.08)
    )

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    if save:
        out_path = os.path.join(save_dir, f"{algo_name}_training_curves.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves for {algo_name.upper()} --> {out_path}")
    else:
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
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    # Aesthetics
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.dpi": 150
    })

    # Unpack histories
    ppo_reward, ppo_fruits, ppo_wall = ppo_data
    a2c_reward, a2c_fruits, a2c_wall = a2c_data
    ddqn_reward, ddqn_fruits, ddqn_wall = ddqn_data

    # Moving average
    kernel = np.ones(window) / window
    smooth = lambda x: np.convolve(x, kernel, mode='valid')

    # Smoothed curves
    ppo_r, ppo_f, ppo_w = smooth(ppo_reward), smooth(ppo_fruits), smooth(ppo_wall)
    a2c_r, a2c_f, a2c_w = smooth(a2c_reward), smooth(a2c_fruits), smooth(a2c_wall)
    ddqn_r, ddqn_f, ddqn_w = smooth(ddqn_reward), smooth(ddqn_fruits), smooth(ddqn_wall)

    colors = {
        "A2C": "#1f77b4",
        "DDQN": "#d62728",
        "PPO": "#2ca02c"
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    # --- REWARD ---
    ax = axes[0]
    ax.axhline(baseline_reward, color="black", linestyle="--", linewidth=1.5)
    ax.plot(a2c_r, label="A2C", color=colors["A2C"], linewidth=2)
    ax.plot(ddqn_r, label="DDQN", color=colors["DDQN"], linewidth=2)
    ax.plot(ppo_r, label="PPO", color=colors["PPO"], linewidth=2)
    ax.set_title("Reward per Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")

    # --- FRUITS ---
    ax = axes[1]
    ax.axhline(baseline_fruits, color="black", linestyle="--", linewidth=1.5)
    ax.plot(a2c_f, color=colors["A2C"], linewidth=2)
    ax.plot(ddqn_f, color=colors["DDQN"], linewidth=2)
    ax.plot(ppo_f, color=colors["PPO"], linewidth=2)
    ax.set_title("Fruits per Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fruits")

    # --- WALL COLLISIONS ---
    ax = axes[2]
    ax.axhline(baseline_wall, color="black", linestyle="--", linewidth=1.5)
    ax.plot(a2c_w, color=colors["A2C"], linewidth=2)
    ax.plot(ddqn_w, color=colors["DDQN"], linewidth=2)
    ax.plot(ppo_w, color=colors["PPO"], linewidth=2)
    ax.set_title("Wall Collisions per Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Collisions")

    # Avoid overlapping x-ticks
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis='x', labelrotation=0)

    # --- SINGLE GLOBAL LEGEND ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout(rect=(0, 0.05, 1, 1))

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

            # PPO AGENT
            if hasattr(model, "net"):
                logits, _ = model.net(state_tensor)
                logits = logits + mask * -1.0
                actions = torch.argmax(logits, dim=1)

            # DDQN AGENT
            elif isinstance(model, DDQNNet):
                s_tensor_nchw = state_tensor.permute(0, 3, 1, 2)
                q_values = model(s_tensor_nchw)
                actions = torch.argmax(q_values, dim=1)

            # A2C AGENT
            else:
                logits, _ = model(state_tensor)
                logits = logits + mask * -1.0
                actions = torch.argmax(logits, dim=1)

        actions_np = actions.cpu().numpy().reshape(-1, 1)

        game_env.move(actions_np)
        state = game_env.to_state()

        frames.append(game_env.boards[0].copy())

    # STATIC SNAPSHOTS
    snapshot_steps = [0, 10, 20, 30, 40]
    snapshot_steps = [s for s in snapshot_steps if s < len(frames)]

    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(15, 3))

    for ax, step in zip(axes, snapshot_steps):
        ax.imshow(frames[step], origin='lower', cmap='viridis', vmin=0, vmax=4)
        ax.set_title(f"{agent_name} – Step {step}")
        ax.axis('off')

    plt.show()

    # ANIMATION
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
