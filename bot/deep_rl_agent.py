import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# ============================
#   DEEP Q-NETWORK (DQN)
# ============================

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# =======================================
#   DEEP RL AGENT WITH EXPERIENCE REPLAY
# =======================================

class DeepRLAgent:
    def __init__(self, state_dim=6, action_dim=3):
        """
        state_dim = features passed from ML model
        action_dim = 3 (LONG, SHORT, HOLD)
        """

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.95
        self.lr = 0.0005
        self.epsilon = 0.15            # exploration probability
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9995

        self.buffer = deque(maxlen=5000)
        self.batch_size = 32

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    # ============================
    #   CHOOSE ACTION
    # ============================

    def get_action(self, state):
        """
        state is a numpy array of ML features:
        [confidence, vol, return, rsi, momentum, trend]
        """

        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        # --- Exploration ---
        if np.random.rand() < self.epsilon:
            choice = np.random.choice(3)
            return choice, 0.1, 1.0  # action, confidence_boost, risk_multiplier

        # --- Exploitation ---
        q_values = self.model(state)
        action = torch.argmax(q_values).item()

        # MAP ACTION
        if action == 0:
            return 0, 0.15, 1.3   # LONG
        elif action == 1:
            return 1, 0.12, 1.4   # SHORT
        else:
            return 2, -0.10, 0.6  # HOLD

    # ============================
    #   STORE EXPERIENCE
    # ============================

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # ============================
    #   TRAIN NETWORK
    # ============================

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        minibatch = random.sample(self.buffer, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_f = target_f.clone().detach()
            target_f[action] = target

            prediction = self.model(state)
            loss = self.loss_fn(prediction, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Lower epsilon gradually
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ============================
    #   TRADE RESULT LEARNING
    # ============================

    def update_after_trade(self, reward):
        """Reward is PnL from trade."""
        reward = float(reward)
        self.train_step()
