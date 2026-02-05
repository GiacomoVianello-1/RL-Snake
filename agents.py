import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from dataclasses import dataclass
from utils import device
from models import PPOActorCritic


# ---- PPO Components ----

# Rollout Buffer for PPO
@dataclass
class RolloutBuffer:
    obs: list
    actions: list
    logprobs: list
    rewards: list
    values: list

    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []

    def add(self, obs, action, logprob, reward, value):
        # obs, reward, value, logprob, action are numpy or scalars
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)

    def to_tensors(self):
        # conver all to torch tensors
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=device)
        logprobs = torch.tensor(np.array(self.logprobs), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        values = torch.tensor(np.array(self.values), dtype=torch.float32, device=device)
        return obs, actions, logprobs, rewards, values

# GAE computation
def compute_gae(rewards, values, last_value, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    advantages = torch.zeros(T, N, device=device)
    gae = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

# PPO Agent (full)
class PPOAgent:
    def __init__(
        self,
        board_size=7,
        n_actions=4,
        lr=4e-4,
        gamma=0.995,
        lam=0.95,
        clip_eps=0.2,
        epochs=4,
        batch_size=1024,
        ent_coef=0.05,   
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.ent_coef_initial = ent_coef

        self.net = PPOActorCritic(board_size=board_size, n_channels=4, n_actions=n_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def update(self, buffer: RolloutBuffer, last_value):
        # Move last_value to device
        last_value = last_value.to(device)

        obs, actions, old_logprobs, rewards, values = buffer.to_tensors()

        # obs: (T, N, H, W, C)
        T, N = rewards.shape
        obs = obs.view(T * N, *obs.shape[2:])
        actions = actions.view(T * N)
        old_logprobs = old_logprobs.view(T * N)
        rewards = rewards.view(T, N)
        values = values.view(T, N)
        last_value = last_value.view(N)

        advantages, returns = compute_gae(
            rewards, values, last_value,
            gamma=self.gamma, lam=self.lam
        )

        advantages = advantages.view(T * N)
        returns = returns.view(T * N)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        dataset_size = T * N
        idxs = np.arange(dataset_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        batches = 0

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                # we select the batch and compute the losses
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                logprobs, entropy, values_pred = self.net.evaluate_actions(batch_obs, batch_actions)

                ratio = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_pred, batch_returns)

                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                # accumulate 
                total_policy_loss += policy_loss.item() 
                total_value_loss += value_loss.item() 
                total_entropy += entropy.mean().item() 
                total_loss += loss.item() 
                batches += 1
        return { 
            "policy_loss": total_policy_loss / batches, 
            "value_loss": total_value_loss / batches, 
            "entropy": total_entropy / batches, 
            "total_loss": total_loss / batches, 
        }
    
# ---- DDQN Components ----

# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next):
        self.buffer.append((s, a, r, s_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=device),
            torch.tensor(np.array(a), dtype=torch.long, device=device),
            torch.tensor(np.array(r), dtype=torch.float32, device=device),
            torch.tensor(np.array(s_next), dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)
