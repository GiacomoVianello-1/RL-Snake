import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import device, make_env, set_seed, save_training_results, get_safety_mask
from models import DDQNNet
from agents import ReplayBuffer

# --- CONFIGURATION ---
NUM_BOARDS = 1000
BOARD_SIZE = 7
ITERATIONS = 20_000_000
SAVE_RESULTS = False

def train_ddqn(
    total_steps=2_000_000,
    n_boards=NUM_BOARDS,
    board_size=7,
    batch_size=256,
    gamma=0.995,
    lr=4e-4,
    start_learning=10_000,
    target_update_interval=5_000,
    eps_start=0.5,
    eps_end=0.01,
):
    env = make_env(n_boards=n_boards, board_size=board_size)
    online = DDQNNet(board_size=board_size).to(device)
    target = DDQNNet(board_size=board_size).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer()

    state = env.to_state()
    step_count = 0
    pbar = tqdm(total=total_steps, desc="Training DDQN")

    rew_hist, fruits_hist, deaths_hist = [], [], []

    last_log_step = 0
    log_interval = 500_000

    # Exponential epsilon decay
    DECAY_STEPS = int(ITERATIONS * (2/3))
    tau = DECAY_STEPS / 3.0

    def epsilon_by_step(t):
        return eps_end + (eps_start - eps_end) * np.exp(-t / tau)

    while step_count < total_steps:
        s = state  # (N, H, W, C)
        eps = epsilon_by_step(step_count)

        # Epsilon-greedy
        with torch.no_grad():
            s_tensor = torch.tensor(s, dtype=torch.float32, device=device).permute(0, 3, 1, 2) # (N, C, H, W)
            q_values = online(s_tensor) # (N, 4)

            # Safety mask
            mask = get_safety_mask(env)
            penalty = -1
            q_values = q_values + penalty * mask

            greedy_actions = torch.argmax(q_values, dim=-1).cpu().numpy()

        random_mask = np.random.rand(n_boards) < eps
        random_actions = np.random.randint(0, 4, size=n_boards)
        actions = np.where(random_mask, random_actions, greedy_actions).reshape(-1, 1)

        rewards = env.move(actions)
        r = rewards.cpu().numpy().flatten()
        next_state = env.to_state()

        # logging
        rew_hist.append(np.mean(r))
        fruits_hist.append(np.sum(r == env.FRUIT_REWARD))
        deaths_hist.append(np.sum(r == env.HIT_WALL_REWARD))

        # push transitions (per board) in buffer
        for i in range(n_boards):
            buffer.push(s[i], actions[i, 0], r[i], next_state[i])

        state = next_state
        step_count += n_boards
        pbar.update(n_boards)

        # learning
        if len(buffer) >= start_learning:
            s_b, a_b, r_b, s_next_b = buffer.sample(batch_size) # (B, H, W, C)
            
            s_b = s_b.permute(0, 3, 1, 2) # (B, C, H, W)
            s_next_b = s_next_b.permute(0, 3, 1, 2)

            # Q(s,a)
            q = online(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)

            # Double DQN target
            with torch.no_grad():
                online_next = online(s_next_b)
                best_actions = torch.argmax(online_next, dim=-1, keepdim=True)
                target_next = target(s_next_b).gather(1, best_actions).squeeze(1)
                y = r_b + gamma * target_next

            loss = F.mse_loss(q, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # target update
        if step_count % target_update_interval < n_boards:
            target.load_state_dict(online.state_dict())

        # logging
        if step_count - last_log_step >= log_interval:
            last_log_step = step_count
            tqdm.write(
                f"\nSteps: {step_count:,}"
                f"\n  Reward (last 500): {np.mean(rew_hist[-500:]):.3f}"
                f"\n  Fruits (last 500): {np.mean(fruits_hist[-500:]):.2f}"
                f"\n  Deaths (last 500): {np.mean(deaths_hist[-500:]):.2f}"
                f"\n  Epsilon: {eps:.3f}"
            )

    pbar.close()
    return online, rew_hist, fruits_hist, deaths_hist

if __name__ == "__main__":
    set_seed(0)
    
    '''
    REMEMBER: adjust the ITERATIONS, NUM_BOARDS, and BOARD_SIZE variables at the top.
    NOTE: saving is disabled by defalut to avoid cluttering the results folder. Change it if needed. 
    '''

    # --- Train DDQN ---
    ddqn_agent, ddqn_rew_hist, ddqn_fruits_hist, ddqn_deaths_hist = train_ddqn(
        total_steps = ITERATIONS,
        n_boards = NUM_BOARDS,
        board_size = BOARD_SIZE
    )

    # Save DDQN results
    save_training_results(
        algo_name="ddqn",
        model=ddqn_agent,
        reward_hist=ddqn_rew_hist,
        fruits_hist=ddqn_fruits_hist,
        deaths_hist=ddqn_deaths_hist,
        save=SAVE_RESULTS
    )