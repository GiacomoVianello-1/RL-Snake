import torch
import numpy as np
import pandas as pd
import pickle
import os

from utils import device
from agents import PPOAgent

# Import BOTH environment types
import environments_fully_observable 
import environments_partially_observable

# CONFIGURATION
NUM_BOARDS = 1000
BOARD_SIZE = 7
EVAL_STEPS = 500  


# ENVIRONMENT CONSTRUCTORS

def get_full_env(n, size):
    """Returns the Allocentric 7x7 Environment"""
    return environments_fully_observable.OriginalSnakeEnvironment(n, size)

def get_partial_env(n, size):
    # Mask size 1 = 3x3 View (Tunnel Vision)
    return environments_partially_observable.OriginalSnakeEnvironment(n, size, mask_size=1)

# BASELINE (Robust to Env Type)
def baseline_policy(env):
    """
    The heuristic baseline works on ANY environment because it reads 'env.boards' directly.
    In the Partial Case, this acts as a 'Perfect Information' control.
    """
    n = env.n_boards
    actions = np.zeros((n, 1), dtype=np.int32)
    heads = np.argwhere(env.boards == env.HEAD)
    fruits = np.argwhere(env.boards == env.FRUIT)
    
    heads = heads[heads[:, 0].argsort()]
    fruits = fruits[fruits[:, 0].argsort()]

    moves = {env.UP: (1, 0), env.DOWN: (-1, 0), env.RIGHT: (0, 1), env.LEFT: (0, -1)}

    for i in range(n):
        _, hx, hy = heads[i]
        _, fx, fy = fruits[i]
        
        # Greedy logic
        if fx > hx: a = env.UP
        elif fx < hx: a = env.DOWN
        elif fy > hy: a = env.RIGHT
        elif fy < hy: a = env.LEFT
        else: a = env.UP

        # Safety Check
        dx, dy = moves[a]
        nx, ny = hx + dx, hy + dy
        if env.boards[i, nx, ny] in (env.BODY, env.WALL):
            safe = []
            for act, (dx, dy) in moves.items():
                tx, ty = hx + dx, hy + dy
                if env.boards[i, tx, ty] not in (env.BODY, env.WALL):
                    safe.append(act)
            a = np.random.choice(safe) if safe else np.random.choice(list(moves.keys()))
        actions[i] = a
    return actions


# EVALUATION LOOP
def run_evaluation(agent, env, steps, name):
    print(f"Evaluating {name}...")
    fruits, deaths, rew_mean = [], [], []

    for _ in range(steps):
        # Get Action
        if agent == "baseline":
            actions = baseline_policy(env)
        else:
            state = env.to_state()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = agent.net(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample().cpu().numpy().reshape(-1, 1)

        # Step Environment
        rewards = env.move(actions)
        if hasattr(rewards, "numpy"): rewards = rewards.numpy()
        rewards = rewards.flatten()

        # Record Metrics
        fruits.append(np.sum(rewards == env.FRUIT_REWARD))
        deaths.append(np.sum(rewards == env.HIT_WALL_REWARD))
        rew_mean.append(np.mean(rewards))

    return [np.mean(rew_mean), np.mean(fruits), np.mean(deaths)]

if __name__ == "__main__":
    results = {}

    # --- LOAD MODELS ---
    
    # PPO FULL (Trained on 7x7)
    try:
        agent_full = PPOAgent(board_size=7) 
        agent_full.net.load_state_dict(torch.load("results/ppo/ppo_weights.pt", map_location=device))
        print("✅ PPO (Full Info) loaded.")
    except:
        agent_full = None
        print("❌ PPO (Full Info) weights not found.")

    # PPO PARTIAL (Trained on 3x3)
    try:
        agent_partial = PPOAgent(board_size=3) 
        # Check standard path or custom path
        path = "results/ppo_partial_3x3/ppo_weights.pt"
        if not os.path.exists(path):
            path = "results/ppo_partial_3x3/ppo_partial_3x3_weights.pt"
            
        agent_partial.net.load_state_dict(torch.load(path, map_location=device))
        print("✅ PPO (Partial Info) loaded.")
    except Exception as e:
        agent_partial = None
        print(f"❌ PPO (Partial Info) weights not found: {e}")

    print("\n==== COMPARATIVE ANALYSIS ====\n")

    # 1. BASELINE in FULL ENV
    # Control: How well does the heuristic work in general?
    env = get_full_env(NUM_BOARDS, BOARD_SIZE)
    results["Baseline (Full Env)"] = run_evaluation("baseline", env, EVAL_STEPS, "Baseline [Full Env]")

    # 2. PPO in FULL ENV
    # Experiment: Can RL beat the heuristic with full info?
    if agent_full:
        env = get_full_env(NUM_BOARDS, BOARD_SIZE)
        results["PPO (Full Env)"] = run_evaluation(agent_full, env, EVAL_STEPS, "PPO [Full Env]")

    # 3. BASELINE in PARTIAL ENV
    # Control: Since Baseline "cheats" and reads env.boards, this should match "Baseline (Full Env)".
    # This confirms that the mechanics of the Partial Env are identical to the Full Env.
    env = get_partial_env(NUM_BOARDS, BOARD_SIZE)
    results["Baseline (Partial Env)"] = run_evaluation("baseline", env, EVAL_STEPS, "Baseline [Partial Env]")

    # 4. PPO in PARTIAL ENV
    # Experiment: How much performance is lost when the Agent is blinded (3x3 view)?
    if agent_partial:
        env = get_partial_env(NUM_BOARDS, BOARD_SIZE)
        results["PPO (Partial Env)"] = run_evaluation(agent_partial, env, EVAL_STEPS, "PPO [Partial Env]")

    # --- FORMATTING OUTPUT ---
    df = pd.DataFrame(results, index=["Reward (Avg)", "Fruits (Avg)", "Wall Hits (Avg)"])
    
    # Transpose for easier reading (Agents as rows)
    df = df.T 
    
    print("\nFinal Performance Comparison (Mean over", EVAL_STEPS, "steps):\n")
    print(df)
    
    # [EXTRA] Save to CSV for latex
    #df.to_csv("results/full_vs_par.csv")
    #print("\n[INFO] Results saved to 'results/full_vs_par.csv'")