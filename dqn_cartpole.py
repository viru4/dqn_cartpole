# dqn_cartpole.py

import gymnasium as gym
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Hyperparameters
# -------------------------------
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000

EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

TARGET_UPDATE = 10
EPISODES = 200

# -------------------------------
# Neural Network
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -------------------------------
# Environment
# -------------------------------
env = gym.make("CartPole-v1", render_mode="human")
# state, _ = env.reset()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# -------------------------------
# Networks
# -------------------------------
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)

memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON

# -------------------------------
# Training Loop
# -------------------------------
for episode in range(EPISODES):

    state, _ = env.reset()
    total_reward = 0

    for t in range(500):

        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state)
            action = torch.argmax(policy_net(state_tensor)).item()

        next_state, reward, done, truncated, _ = env.step(action)

        memory.store((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        # Train
        if len(memory) > BATCH_SIZE:

            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))

            # Current Q-values
            q_values = policy_net(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Target Q-values
            next_q_values = target_net(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q_values * (1 - dones)

            # Loss
            loss = nn.MSELoss()(q_values, targets.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done or truncated:
            break

    # Update epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()