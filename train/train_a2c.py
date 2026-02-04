import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils import device, make_env, set_seed, save_training_results
from models import A2CNet

# --- CONFIGURATION ---
NUM_BOARDS = 1000
BOARD_SIZE = 7
ITERATIONS = 2_000_000
SAVE_RESULTS = False

def train_a2c(total_steps, n_boards, board_size):
    env = make_env(n_boards=n_boards, board_size=board_size)
    net = A2CNet(board_size=board_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    state = env.to_state()
    step_count = 0
    pbar = tqdm(total=total_steps, desc="Training A2C")
    gamma = 0.99

    reward_hist, fruits_hist, deaths_hist = [], [], []
    last_log_step = 0
    log_interval = 500_000

    while step_count < total_steps:
        s = torch.tensor(state, dtype=torch.float32, device=device)
        logits, v_s = net(s)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()

        rewards = env.move(a.cpu().numpy().reshape(-1,1))
        r = rewards.cpu().numpy().flatten()
        next_state = env.to_state()

        with torch.no_grad():
            s_next = torch.tensor(next_state, dtype=torch.float32, device=device)
            _, v_next = net(s_next)

        r_t = torch.tensor(r, dtype=torch.float32, device=device)
        delta = r_t + gamma * v_next - v_s

        logprobs = dist.log_prob(a)
        policy_loss = -(logprobs * delta.detach()).mean()
        value_loss = delta.pow(2).mean()
        entropy = dist.entropy().mean()

        loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        step_count += n_boards
        pbar.update(n_boards)

        reward_hist.append(np.mean(r))
        fruits_hist.append(np.sum(r == env.FRUIT_REWARD))
        deaths_hist.append(np.sum(r == env.HIT_WALL_REWARD))

        if step_count - last_log_step >= log_interval:
            last_log_step = step_count
            tqdm.write(f"Steps: {step_count:,} | Reward: {np.mean(reward_hist[-500:]):.3f}")

    pbar.close()
    return net, reward_hist, fruits_hist, deaths_hist

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