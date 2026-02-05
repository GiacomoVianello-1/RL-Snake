import torch
import numpy as np
import pandas as pd
import pickle

from utils import device, get_env, make_env, plot_training_curves, plot_comparison_curves, display_game
from models import A2CNet, DDQNNet
from agents import PPOAgent

NUM_BOARDS = 1000
BOARD_SIZE = 7
EVAL_STEPS = 100
SAVE_PLOTS = True

# ===============
# BASELINE POLICY
# ===============
def baseline_policy(env):
    n = env.n_boards
    actions = np.zeros((n, 1), dtype=np.int32)

    heads = np.argwhere(env.boards == env.HEAD)
    fruits = np.argwhere(env.boards == env.FRUIT)

    heads = heads[heads[:, 0].argsort()]
    fruits = fruits[fruits[:, 0].argsort()]

    moves = {
        env.UP:    (1, 0),
        env.DOWN:  (-1, 0),
        env.RIGHT: (0, 1),
        env.LEFT:  (0, -1)
    }

    for i in range(n):
        _, hx, hy = heads[i]
        _, fx, fy = fruits[i]

        # greedy direction toward fruit
        if fx > hx: a = env.UP
        elif fx < hx: a = env.DOWN
        elif fy > hy: a = env.RIGHT
        elif fy < hy: a = env.LEFT
        else: a = env.UP

        # compute next head position
        dx, dy = moves[a]
        nx, ny = hx + dx, hy + dy

        # check BODY or WALL
        if env.boards[i, nx, ny] in (env.BODY, env.WALL):
            safe = []
            for act, (dx, dy) in moves.items():
                tx, ty = hx + dx, hy + dy
                if env.boards[i, tx, ty] not in (env.BODY, env.WALL):
                    safe.append(act)

            if len(safe) == 0:
                a = np.random.choice([env.UP, env.DOWN, env.LEFT, env.RIGHT])
            else:
                a = np.random.choice(safe)

        actions[i] = a

    return actions

# Evaluate baseline policy
def evaluate_baseline(steps=1000):
    env = get_env(n=NUM_BOARDS)
    fruits = []
    deaths = []
    reward_means = []

    for _ in range(steps):
        actions = baseline_policy(env)
        rewards = env.move(actions)
        rewards = rewards.numpy().flatten() if hasattr(rewards, "numpy") else rewards

        # metriche
        fruits.append(np.sum(rewards == env.FRUIT_REWARD))
        deaths.append(np.sum(rewards == env.HIT_WALL_REWARD))
        reward_means.append(np.mean(rewards))

    return (
        np.mean(np.array(fruits)),
        np.mean(np.array(deaths)),
        np.mean(np.array(reward_means))
    )

def evaluate_agent(agent, steps=1000):
    env = get_env(n=NUM_BOARDS)
    fruits = []
    deaths = []
    rewards_mean = []

    for _ in range(steps):
        state = env.to_state()  # (N, H, W, C)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():

            # PPO agent --> has .net
            if hasattr(agent, "net"):
                logits, _ = agent.net(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample().cpu().numpy().reshape(-1, 1)

            # DDQN agent --> class is DDQNNet
            elif isinstance(agent, DDQNNet):
                s_tensor_nchw = state_tensor.permute(0, 3, 1, 2)
                q_values = agent(s_tensor_nchw)
                actions = torch.argmax(q_values, dim=-1).cpu().numpy().reshape(-1, 1)

            # A2C agent --> returns (logits, value)
            else:
                logits, _ = agent(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample().cpu().numpy().reshape(-1, 1)

        rewards = env.move(actions).cpu().numpy().flatten()

        fruits.append(np.sum(rewards == env.FRUIT_REWARD))
        deaths.append(np.sum(rewards == env.HIT_WALL_REWARD))
        rewards_mean.append(np.mean(rewards))

    return (
        np.array(fruits),
        np.array(deaths),
        np.array(rewards_mean)
    )

if __name__ == "__main__":

    results = {}

    # --- LOAD MODELS ---

    # DDQN
    try:
        ddqn_agent = DDQNNet(board_size=BOARD_SIZE).to(device)
        ddqn_agent.load_state_dict(torch.load("results/ddqn/ddqn_weights.pt"))
        with open("results/ddqn/ddqn_training.pkl", "rb") as f:
            ddqn_data = pickle.load(f)
        ddqn_reward_hist = ddqn_data["reward_hist"]
        ddqn_fruits_hist = ddqn_data["fruits_hist"]
        ddqn_wall_hist = ddqn_data["deaths_hist"]
        print("✅ DDQN weights and training history loaded.")
    except Exception as e:
        print(f"❌ Failed to load DDQN: {e}")
        ddqn_agent = None
    
    # A2C
    try:
        a2c_agent = A2CNet(board_size=BOARD_SIZE).to(device)
        a2c_agent.load_state_dict(torch.load("results/a2c/a2c_weights.pt"))
        with open("results/a2c/a2c_training.pkl", "rb") as f:
            a2c_data = pickle.load(f)
        a2c_reward_hist = a2c_data["reward_hist"]
        a2c_fruits_hist = a2c_data["fruits_hist"]
        a2c_wall_hist = a2c_data["deaths_hist"]
        print("✅ A2C weights and training history loaded.")
    except Exception as e:
        print(f"❌ Failed to load A2C: {e}")
        a2c_agent = None

    # PPO
    try:
        ppo_agent = PPOAgent(board_size=BOARD_SIZE)
        ppo_agent.net.load_state_dict(torch.load("results/ppo/ppo_weights.pt", map_location=device))
        with open("results/ppo/ppo_training.pkl", "rb") as f:
            ppo_data = pickle.load(f)
        ppo_reward_hist = ppo_data["reward_hist"]
        ppo_fruits_hist = ppo_data["fruits_hist"]
        ppo_wall_hist = ppo_data["deaths_hist"]
        print("✅ PPO weights and training history loaded.")
    except Exception as e:
        print(f"❌ Failed to load PPO: {e}")
        ppo_agent = None

    print("\n ==== EVALUATION ====\n")

    
    # Additional Baseline Evaluation Table (SKIP as it requires so much time to run...)
    '''
    print("\n--- Baseline Evaluation at Multiple Horizons ---")

    baseline_steps = [100, 1000, 10000]
    baseline_results = {}

    for steps in baseline_steps:
        print(f"Evaluating baseline for {steps} steps...")
        b_fruit, b_wall, b_rew = evaluate_baseline(steps)
        baseline_results[f"{steps} steps"] = [
            np.mean(b_rew),
            np.mean(b_fruit),
            np.mean(b_wall)
        ]

    df_baseline = pd.DataFrame(
        baseline_results,
        index=["Reward (Avg)", "Fruits (Avg)", "Wall hits (Avg)"]
    )

    print("\nBaseline Performance Across Different Evaluation Horizons:\n")
    print(df_baseline)
    print("\n")
    '''

    print("** Running Baseline **")
    b_fruit, b_wall, b_rew = evaluate_baseline(EVAL_STEPS)
    results["Baseline"] = [b_rew, b_fruit, b_wall]

    if a2c_agent:
        print("** Running A2C **")
        a_fruit, a_wall, a_rew = evaluate_agent(a2c_agent, EVAL_STEPS)
        results["A2C"] = [np.mean(a_rew), np.mean(a_fruit), np.mean(a_wall)]

    if ddqn_agent:
        print("** Running DDQN **")
        d_fruit, d_wall, d_rew = evaluate_agent(ddqn_agent, EVAL_STEPS)
        results["DDQN"] = [np.mean(d_rew), np.mean(d_fruit), np.mean(d_wall)]
    
    if ppo_agent:
        print("** Running PPO **")
        p_fruit, p_wall, p_rew = evaluate_agent(ppo_agent, EVAL_STEPS)
        results["PPO"] = [np.mean(p_rew), np.mean(p_fruit), np.mean(p_wall)]
    
    # DataFrame 
    df = pd.DataFrame(results, index=["Reward (Avg)", "Fruits (Avg)", "Wall hits (Avg)"]) 
    print("\nEvaluation Results over", EVAL_STEPS, "steps per agent (using pretrained models):\n") 
    print(df)
    print("\n")

    # --- Training Results ---
    plot_training_curves(
        algo_name="a2c",
        reward_hist=a2c_reward_hist,
        fruits_hist=a2c_fruits_hist,
        wall_hist=a2c_wall_hist,
        baseline_reward=b_rew,
        baseline_fruits=b_fruit,
        baseline_wall=b_wall,
        save=SAVE_PLOTS
    )
    plot_training_curves(
        algo_name="ddqn",
        reward_hist=ddqn_reward_hist,
        fruits_hist=ddqn_fruits_hist,
        wall_hist=ddqn_wall_hist,
        baseline_reward=b_rew,
        baseline_fruits=b_fruit,
        baseline_wall=b_wall,
        save=SAVE_PLOTS
    )
    plot_training_curves(
        algo_name="ppo",
        reward_hist=ppo_reward_hist,
        fruits_hist=ppo_fruits_hist,
        wall_hist=ppo_wall_hist,
        baseline_reward=b_rew,
        baseline_fruits=b_fruit,
        baseline_wall=b_wall,
        save=SAVE_PLOTS
    )

    # Full comparison
    plot_comparison_curves(
        ppo_data=(ppo_reward_hist, ppo_fruits_hist, ppo_wall_hist),
        a2c_data=(a2c_reward_hist, a2c_fruits_hist, a2c_wall_hist),
        ddqn_data=(ddqn_reward_hist, ddqn_fruits_hist, ddqn_wall_hist),
        baseline_reward=b_rew,
        baseline_fruits=b_fruit,
        baseline_wall=b_wall,
        save=SAVE_PLOTS,
    )

    # --- Play agent ---
    display_game(a2c_agent, agent_name="A2C", max_steps=100)
    display_game(ddqn_agent, agent_name="DDQN", max_steps=100)
    display_game(ppo_agent, agent_name="PPO", max_steps=100)
