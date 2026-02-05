import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from utils import device, make_env, set_seed, save_training_results, get_safety_mask
from models import A2CNet

# --- CONFIGURATION ---
NUM_BOARDS = 1000
BOARD_SIZE = 7
ITERATIONS = 20_000_000
SAVE_RESULTS = False

def train_a2c(total_steps=2_000_000, n_boards=NUM_BOARDS, board_size=7):
    env = make_env(n_boards=n_boards, board_size=board_size)
    net = A2CNet(board_size=board_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    state = env.to_state()
    step_count = 0
    pbar = tqdm(total=total_steps, desc="Training A2C")

    gamma = 0.995

    # Logging buffers
    reward_history = []
    fruits_history = []
    deaths_history = []
    policy_loss_hist = []
    value_loss_hist = []
    entropy_hist = []

    last_log_step = 0
    log_interval = 750_000

    while step_count < total_steps:
        s = torch.tensor(state, dtype=torch.float32, device=device)

        # Forward pass
        logits, v_s = net(s)

        # Safety Mask
        mask = get_safety_mask(env)
        mask_tensor = mask.to(device)
        penalty = -1
        logits = logits + penalty * mask_tensor

        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()

        # Step environment
        rewards = env.move(a.cpu().numpy().reshape(-1,1))
        r = rewards.cpu().numpy().flatten()
        next_state = env.to_state()

        # Bootstrap V(s')
        with torch.no_grad():
            s_next = torch.tensor(next_state, dtype=torch.float32, device=device)
            _, v_next = net(s_next)

        # TD error δ = r + γV(s') − V(s)
        r_t = torch.tensor(r, dtype=torch.float32, device=device)
        delta = r_t + gamma * v_next - v_s

        # Losses
        logprobs = dist.log_prob(a)
        policy_loss = -(logprobs * delta.detach()).mean()
        value_loss = delta.pow(2).mean()
        entropy = dist.entropy().mean()

        loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state
        state = next_state
        step_count += n_boards
        pbar.update(n_boards)

        # Store metrics
        reward_history.append(np.mean(r))
        fruits_history.append(np.sum(r == env.FRUIT_REWARD))
        deaths_history.append(np.sum(r == env.HIT_WALL_REWARD))
        policy_loss_hist.append(policy_loss.item())
        value_loss_hist.append(value_loss.item())
        entropy_hist.append(entropy.item())

        # Periodic logging
        if step_count - last_log_step >= log_interval:
            last_log_step = step_count
            tqdm.write(
                f"\nSteps: {step_count:,}"
                f"\n  Reward (last 500): {np.mean(reward_history[-500:]):.3f}"
                f"\n  Fruits (last 500): {np.mean(fruits_history[-500:]):.2f}"
                f"\n  Deaths (last 500): {np.mean(deaths_history[-500:]):.2f}"
                f"\n  Policy Loss (last 500): {np.mean(policy_loss_hist[-500:]):.4f}"
                f"\n  Value Loss  (last 500): {np.mean(value_loss_hist[-500:]):.4f}"
                f"\n  Entropy     (last 500): {np.mean(entropy_hist[-500:]):.4f}"
            )

    pbar.close()

    return net, reward_history, fruits_history, deaths_history

if __name__ == "__main__":
    set_seed(0)

    '''
    REMEMBER: adjust the ITERATIONS, NUM_BOARDS, and BOARD_SIZE variables at the top.
    NOTE: saving is disabled by defalut to avoid cluttering the results folder. Change it if needed. 
    '''

    print("Starting A2C Training...")
    a2c_agent, a2c_reward_hist, a2c_fruits_hist, a2c_deaths_hist = train_a2c(
        total_steps=ITERATIONS,
        n_boards=NUM_BOARDS,
        board_size=BOARD_SIZE
    )

    # Save A2C results
    save_training_results(
        algo_name="a2c",
        model=a2c_agent,
        reward_hist=a2c_reward_hist,
        fruits_hist=a2c_fruits_hist,
        deaths_hist=a2c_deaths_hist,
        save=SAVE_RESULTS
    )