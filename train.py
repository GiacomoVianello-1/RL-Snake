import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pickle
from tqdm import tqdm

from utils import device, make_env, get_safety_mask, set_seed
from models import A2CNet, DDQNNet
from agents import PPOAgent, RolloutBuffer, ReplayBuffer

# GLOBAL PARAMETERS
NUM_BOARDS = 1000       # Number of parallel boards to simulate
BOARD_SIZE = 7          # Size of each board (including borders)
ITERATIONS = 10000000   # Total training steps (a lot, but we use torch and GPU support for this reason!)
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

# =====================
# A2C TRAINING FUNCTION
# =====================

def train_a2c(total_steps=2_000_000, n_boards=NUM_BOARDS, board_size=7):
    env = make_env(n_boards=n_boards, board_size=board_size)
    net = A2CNet(board_size=board_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    state = env.to_state()
    step_count = 0
    pbar = tqdm(total=total_steps, desc="Training A2C")

    gamma = 0.99

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
    print(f"[INFO] Training completed for A2C.")
    return net, reward_history, fruits_history, deaths_history

# ======================
# DDQN TRAINING FUNCTION
# ======================

def train_ddqn(
    total_steps=2_000_000,
    n_boards=NUM_BOARDS,
    board_size=7,
    batch_size=256,
    gamma=0.99,
    lr=4e-4,
    start_learning=10_000,
    target_update_interval=5_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=500_000,
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

    rew_hist = []
    fruits_hist = []
    deaths_hist = []

    last_log_step = 0
    log_interval = 500_000

    def epsilon_by_step(t):
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    while step_count < total_steps:
        s = state  # (N, H, W, C)
        eps = epsilon_by_step(step_count)

        # ε-greedy
        with torch.no_grad():
            q_values = online(torch.tensor(s, dtype=torch.float32, device=device))
            greedy_actions = torch.argmax(q_values, dim=-1).cpu().numpy()

        random_mask = np.random.rand(n_boards) < eps
        random_actions = np.random.randint(0, 4, size=n_boards)
        actions = np.where(random_mask, random_actions, greedy_actions).reshape(-1, 1)

        rewards = env.move(actions)
        r = rewards.cpu().numpy().flatten()
        next_state = env.to_state()

        # log per-step metrics
        rew_hist.append(np.mean(r))
        fruits_hist.append(np.sum(r == env.FRUIT_REWARD))
        deaths_hist.append(np.sum(r == env.HIT_WALL_REWARD))

        # push transitions (per board) in buffer
        for i in range(n_boards):
            buffer.push(s[i], actions[i, 0], r[i], next_state[i])

        state = next_state
        step_count += n_boards
        pbar.update(n_boards)

        # learn
        if len(buffer) >= start_learning:
            s_b, a_b, r_b, s_next_b = buffer.sample(batch_size)

            # Q(s,a)
            q = online(s_b).gather(1, a_b.view(-1,1)).squeeze(1)

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
    print(f"[INFO] Training completed for DDQN.")
    return online, rew_hist, fruits_hist, deaths_hist

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
        print(f"[INFO] Saving disabled -> skipping save for {algo_name}")
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

if __name__ == "__main__":
    set_seed(0)

    '''
    SUGGESTION: select one algorithm at a time to train, as training all three sequentially can take hours (especially on GPU)
                comment/uncomment the sections below to choose which algorithm to train.

    RECOMMENDED: train PPO as it generally outperforms A2C and DDQN in this environment.

    REMEMBER: adjust the ITERATIONS, NUM_BOARDS, and BOARD_SIZE variables at the top.

    NOTA: saving is disabled by defalut to avoid cluttering the results folder. Change it if needed. 
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

    # --- Train A2C ---
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