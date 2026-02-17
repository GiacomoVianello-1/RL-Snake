import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pickle
from tqdm import tqdm

import environments_partially_observable 
from utils import device, get_safety_mask, set_seed, save_training_results
from agents import PPOAgent, RolloutBuffer

# GLOBAL PARAMETERS
NUM_BOARDS = 1000       
BOARD_SIZE = 7          
ITERATIONS = 20_000_000 
SAVE_RESULTS = False     


# ENV FACTORY
def make_3x3_env(n_boards, board_size):
    # mask_size=1 creates a 3x3 window around the head.
    return environments_partially_observable.OriginalSnakeEnvironment(
        n_boards, 
        board_size, 
        mask_size=1  # USE A 3x3 grid centered in the snake's head
    )

# TRAINING LOOP
def train_ppo(total_steps=2_000_000, n_boards=256, board_size=7, rollout_horizon=256):
    
    # --- CONFIGURATION FOR 3x3 VIEW ---
    MASK_RADIUS = 1
    OBS_DIM = (MASK_RADIUS * 2) + 1  # Result: 3
    
    # The World (Real Size = 7)
    env = make_3x3_env(n_boards, board_size)
    
    # The Agent (Input Size = 3)
    # We lie to the agent and say "board_size=3". 
    # This forces the CNN to create the correct shapes for 3x3 inputs.
    agent = PPOAgent(board_size=OBS_DIM)

    state = env.to_state()
    step_count = 0

    reward_history, fruits_history, wall_deaths_history, loss_history = [], [], [], []
    last_log_step = 0
    log_interval = 500_000

    pbar = tqdm(total=total_steps, desc="Training PPO (3x3 View)")
    while step_count < total_steps:
        buffer = RolloutBuffer()

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

            # Logging metrics
            reward_history.append(np.mean(rewards_np))
            fruits_history.append(np.sum(rewards_np == env.FRUIT_REWARD))
            wall_deaths_history.append(np.sum(rewards_np == env.HIT_WALL_REWARD))

            buffer.add(state, actions_np.squeeze(-1), logprobs.cpu().numpy(), rewards_np, values.cpu().numpy())
            state = next_state
            step_count += n_boards
            pbar.update(n_boards)
            if step_count >= total_steps: break

        # Update
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad(): _, last_values = agent.net.forward(state_tensor)
        
        loss_dict = agent.update(buffer, last_values) 
        loss_history.append(loss_dict)

        # Entropy decay
        agent.ent_coef = agent.ent_coef_initial * (1 - (step_count / total_steps))

        if step_count - last_log_step >= log_interval:
            last_log_step = step_count
            tqdm.write(
                f"\nSteps: {step_count:,} | Reward: {np.mean(reward_history[-500:]):.3f} | "
                f"Fruits: {np.mean(fruits_history[-500:]):.2f} | "
                f"Deaths: {np.mean(wall_deaths_history[-500:]):.2f}"
            )
            
    pbar.close()
    return agent, reward_history, fruits_history, wall_deaths_history, loss_history

if __name__ == "__main__":
    set_seed(0)

    # Train
    agent, rew, fruit, wall, loss = train_ppo(ITERATIONS, NUM_BOARDS, BOARD_SIZE, 256)
    
    # Save with specific name
    save_training_results(
        algo_name="ppo_partial_3x3",  # Distinct name to avoid overriding
        model=agent,
        reward_hist=rew,
        fruits_hist=fruit,
        deaths_hist=wall,
        extra_metrics={"loss_hist": loss},
        save=SAVE_RESULTS
    )