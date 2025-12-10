import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ============================
# TRANSFORMER MODEL
# ============================

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, output_dim=3):
        super().__init__()

        self.embedding = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]   # use last timestep
        return self.fc(x)


# ============================
# TRANSFORMER DEEP-RL AGENT
# ============================

class TransformerRLAgent:
    """
    ACTION SPACE:
    0 = HOLD
    1 = LONG (BUY)
    2 = SHORT (SELL)
    """

    def __init__(self, sequence_length=30, features=6):
        self.sequence_length = sequence_length
        self.features = features

        self.model = TransformerModel(
            input_dim=features,
            output_dim=3
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=50_000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        self.batch_size = 64

    # ============================
    # STATE BUILDER
    # ============================

    def build_state(self, df):
        """
        Creates a rolling state window for the Transformer.
        """
        window = df.tail(self.sequence_length)

        state = np.column_stack([
            window["Close"].pct_change().fillna(0),
            window["High"] / window["Low"],
            window["Volume"].pct_change().fillna(0),
            window["Close"] / window["Close"].rolling(10).mean(),
            window["Close"] / window["Close"].rolling(30).mean(),
            window["Close"].rolling(10).std().fillna(0)
        ])

        state = np.nan_to_num(state)
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # ============================
    # ACTION SELECTION
    # ============================

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    # ============================
    # MEMORY
    # ============================

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # ============================
    # TRAINING STEP
    # ============================

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        next_q_values = self.model(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_selected, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
