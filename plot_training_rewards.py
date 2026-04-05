# plot_rewards.py

import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(file_path="rewards.npy", window=10):
    rewards = np.load(file_path)

    plt.figure(figsize=(10, 5))

    plt.plot(rewards, label="Episode Reward")

    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f"{window}-Episode Avg")

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("DQN Training Performance (CartPole)")
    plt.legend()
    plt.grid()

    plt.show()

if __name__ == "__main__":
    plot_rewards()