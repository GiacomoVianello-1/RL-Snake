import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple

# ===================================
# Define the PPO Actor-Critic Network
# ===================================

class PPOActorCritic(nn.Module):
    def __init__(self, board_size=7, n_channels=4, n_actions=4):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.board_size = board_size
        conv_out_dim = 128 * board_size * board_size

        self.fc = nn.Linear(conv_out_dim, 128)
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 3, 1, 2)          # (N, H, W, C) -> (N, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))

        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, x, mask=None):
        logits, value = self.forward(x)
        if mask is not None:
            logits = logits + mask
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value


    def evaluate_actions(self, x, actions):
        logits, values = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy, values

# ==================================
# Advantage Actor-Critic (A2C) Agent
# ==================================

class A2CNet(nn.Module):
    def __init__(self, board_size=7, n_channels=4, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        flat = board_size * board_size * 64
        self.policy = nn.Linear(flat, n_actions)
        self.value = nn.Linear(flat, 1)

    def forward(self, x):
        x = x.permute(0,3,1,2)  # NHWC → NCHW
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.policy(x), self.value(x).squeeze(-1)
    
# ==================
# Double DQN Network
# ==================

class DDQNNet(nn.Module):
    def __init__(self, board_size=7, n_channels=4, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        flat = board_size * board_size * 64
        self.q_head = nn.Linear(flat, n_actions)

    def forward(self, x):
        x = x.permute(0,3,1,2)  # NHWC → NCHW
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.q_head(x)   # (B, n_actions)