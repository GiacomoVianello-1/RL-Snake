import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pickle
from tqdm import tqdm

from utils import device, make_env, get_safety_mask, set_seed, save_training_results
from models import A2CNet, DDQNNet
from agents import PPOAgent, RolloutBuffer, ReplayBuffer

# GLOBAL PARAMETERS
NUM_BOARDS = 1000       # Number of parallel boards to simulate
BOARD_SIZE = 7          # Size of each board (including borders)
ITERATIONS = 20_000_000 # Total training steps (a lot, but we use torch and GPU support for this reason!)
SAVE_RESULTS = False    # Whether to save the trained models and results

# =====================
# PPO TRAINING FUNCTION
# =====================

def train_ppo(total_steps=2_000_000, n_boards=256, board_size=7, rollout_horizon=256):
    env = make_env(n_boards=n_boards, board_size=board_size)
    agent = PPOAgent(board_size=board_size)

    state = env.to_state()
    step_count = 0

    reward_history = []
    fruits_history = []
    wall_deaths_history = []
    loss_history = []

    last_log_step = 0
    log_interval = 500000

    pbar = tqdm(total=total_steps, desc="Training PPO")
    while step_count < total_steps:
        buffer = RolloutBuffer()

        # Rollout collection
        for t in range(rollout_horizon):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            mask_tensor = get_safety_mask(env)
            penalty = -1.0

            with torch.no_grad():
                logits, values = agent.net.forward(state_tensor)
                logits = logits + penalty * mask_tensor
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy().reshape(-1, 1)
            rewards_tensor = env.move(actions_np)
            rewards_np = rewards_tensor.cpu().numpy().flatten()

            next_state = env.to_state()

            # metrics
            reward_history.append(np.mean(rewards_np))
            fruits_history.append(np.sum(rewards_np == env.FRUIT_REWARD))
            wall_deaths_history.append(np.sum(rewards_np == env.HIT_WALL_REWARD))

            buffer.add(
                obs=state,
                action=actions_np.squeeze(-1),
                logprob=logprobs.cpu().numpy(),
                reward=rewards_np,
                value=values.cpu().numpy()
            )

            state = next_state
            step_count += n_boards
            pbar.update(n_boards)

            if step_count >= total_steps:
                break

        # PPO update
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, last_values = agent.net.forward(state_tensor)

        loss_dict = agent.update(buffer, last_values) 
        loss_history.append(loss_dict)

        # Entropy decay
        progress = step_count / total_steps
        agent.ent_coef = agent.ent_coef_initial * (1 - progress)

        # Logging every 50k steps
        if step_count - last_log_step >= log_interval:
            last_log_step = step_count

            avg_reward = np.mean(reward_history[-500:])
            avg_fruits = np.mean(fruits_history[-500:])
            avg_deaths = np.mean(wall_deaths_history[-500:])

            avg_policy_loss = np.mean([l["policy_loss"] for l in loss_history[-50:]])
            avg_value_loss = np.mean([l["value_loss"] for l in loss_history[-50:]])
            avg_entropy = np.mean([l["entropy"] for l in loss_history[-50:]])
            avg_total_loss = np.mean([l["total_loss"] for l in loss_history[-50:]])

            tqdm.write(
                f"\nSteps: {step_count:,}"
                f"\n  Reward (last 500): {avg_reward:.3f}"
                f"\n  Fruits (last 500): {avg_fruits:.2f}"
                f"\n  Deaths (last 500): {avg_deaths:.2f}"
                f"\n  Policy Loss (avg 50 updates): {avg_policy_loss:.4f}"
                f"\n  Value Loss  (avg 50 updates): {avg_value_loss:.4f}"
                f"\n  Entropy     (avg 50 updates): {avg_entropy:.4f}"
                f"\n  Total Loss  (avg 50 updates): {avg_total_loss:.4f}"
            )
    pbar.close()
    print(f"[INFO] Training completed for PPO.")
    return agent, reward_history, fruits_history, wall_deaths_history, loss_history

if __name__ == "__main__":
    set_seed(0)

    '''
    REMEMBER: adjust the ITERATIONS, NUM_BOARDS, and BOARD_SIZE variables at the top.
    NOTE: saving is disabled by defalut to avoid cluttering the results folder. Change it if needed. 
    '''

    # --- Train PPO ---
    ppo_agent, ppo_reward_hist, ppo_fruits_hist, ppo_wall_hist, ppo_loss_hist = train_ppo(
        total_steps=ITERATIONS,
        n_boards=NUM_BOARDS,
        board_size=BOARD_SIZE,
        rollout_horizon=256
    )
    # Save PPO results
    save_training_results(
        algo_name="ppo",
        model=ppo_agent,
        reward_hist=ppo_reward_hist,
        fruits_hist=ppo_fruits_hist,
        deaths_hist=ppo_wall_hist,
        extra_metrics={"loss_hist": ppo_loss_hist},
        save=SAVE_RESULTS
    )